# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import shutil
import subprocess
from os import path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from progressbar import progressbar
from torch.utils.data import DataLoader

from inference.data.mask_mapper import MaskMapper
from inference.data.test_datasets import LongTestDataset
from inference.inference_core import InferenceCore
from model.network import XMem

try:
    import hickle as hkl
except ImportError:
    print('Failed to import hickle. Fine if not using multi-scale testing.')


from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        torch.hub.set_dir("./models/hub")
        self.model = "./saves/XMem.pth"
        
    
        """
        Arguments loading
        """

        self.model = "./saves/XMem.pth"
        self.in_path = "./inputs/"
        self.vid_name = "default_video"
        self.out_path = "./results/"
        self.outvideo_path = "result.mp4"

        #self.max_mid_term_frames = 10 # T_max in paper, decrease to save memory
        #self.min_mid_term_frames = 5 # T_min in paper, decrease to save memory
        #self.max_long_term_elements = 10000 # LT_max in paper, increase if objects disappear for a long time
        #self.num_prototypes = 128 # P in paper
        #self.top_k = 30
        #self.mem_every = 5 # r in paper. Increase to improve running speed.
        #self.deep_update_every = -1 # Leave -1 normally to synchronize with mem_every
        self.size = 480 # Resize the shorter side to this size. -1 to use original resolution.
        self.save_all = True
        self.benchmark = None
        
        self.config = {
            'max_mid_term_frames' : 10,
            'min_mid_term_frames' : 5,
            'max_long_term_elements': 10000,
            'num_prototypes' : 128,
            'top_k' : 30,
            'mem_every': -1,
            'deep_update_every' : -1,
            'size' : self.size,
            'disable_long_term': None,
            'enable_long_term': not None,
        }

        torch.autograd.set_grad_enabled(False)

        # Load our checkpoint
        self.network = XMem(self.config, self.model).cuda().eval()
        model_weights = torch.load(self.model)
        self.network.load_weights(model_weights, init_as_zero_if_needed=True)


    def predict(
        self,
        video: Path = Input(description="Source video for object segmentation"),
        mask: Path = Input(description="Segmentation mask for the first frame of the video, of the object(s) we want to segment.")
    ) -> Path:
        """Run a single prediction on the model"""
        # clean up earlier files if they exist
        if Path(self.in_path).exists():
            shutil.rmtree(self.in_path)
        if Path(self.out_path).exists():
            shutil.rmtree(self.out_path)
        if Path(self.outvideo_path).exists():
            os.unlink(self.outvideo_path)


        # save the video and image in the proper folder structure
        video_path = Path(self.in_path) / "JPEGImages" / self.vid_name
        mask_path = Path(self.in_path) / "Annotations" / self.vid_name
        
        os.makedirs(video_path, exist_ok=True)
        os.makedirs(mask_path, exist_ok=True)

        mask_img = Image.open(str(mask))            
        mask_img.save(mask_path / "0001.png")

        # turn the video into a list of frames        
        vidcap = cv2.VideoCapture(str(video))
        success, image = vidcap.read()
        count = 1
        while success:
            cv2.imwrite(str(video_path / f"{count:04d}.jpg"), image)     # save frame as JPEG file      
            success, image = vidcap.read()
            count += 1
        
        # Set up data loader
        meta_dataset = LongTestDataset(path.join(self.in_path), size=self.size)
        meta_loader = meta_dataset.get_datasets()

        # Start eval
        #for vid_reader in progressbar(meta_loader, max_value=len(meta_dataset), redirect_stdout=True):
        for vid_reader in meta_loader:

            loader = DataLoader(vid_reader, batch_size=1, shuffle=False, num_workers=2)
            vid_name = vid_reader.vid_name
            vid_length = len(loader)
            
            # no need to count usage for LT if the video is not that long anyway
            self.config['enable_long_term_count_usage'] = (
                self.config['enable_long_term'] and
                (vid_length
                    / (self.config['max_mid_term_frames']-self.config['min_mid_term_frames'])
                    * self.config['num_prototypes'])
                >= self.config['max_long_term_elements']
            )

            mapper = MaskMapper()
            processor = InferenceCore(self.network, config=self.config)
            first_mask_loaded = False

            for ti, data in progressbar(enumerate(loader), max_value=len(loader)):
                with torch.cuda.amp.autocast(enabled=not self.benchmark):
                    rgb = data['rgb'].cuda()[0]
                    msk = data.get('mask')
                    info = data['info']
                    frame = info['frame'][0]
                    shape = info['shape']
                    need_resize = info['need_resize'][0]

                    if not first_mask_loaded:
                        if msk is not None:
                            first_mask_loaded = True
                        else:
                            # no point to do anything without a mask
                            raise ValueError("There is no mask")
                            continue

                    # Map possibly non-continuous labels to continuous ones
                    if msk is not None:
                        msk, labels = mapper.convert_mask(msk[0].numpy())
                        msk = torch.Tensor(msk).cuda()
                        if need_resize:
                            msk = vid_reader.resize_mask(msk.unsqueeze(0))[0]
                        processor.set_all_labels(list(mapper.remappings.values()))
                    else:
                        labels = None

                    # Run the model on this frame
                    prob = processor.step(rgb, msk, labels, end=(ti==vid_length-1))

                    # Upsample to original size if needed
                    if need_resize:
                        prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:,0]

                    torch.cuda.synchronize()

                    # Probability mask -> index mask
                    out_mask = torch.max(prob, dim=0).indices
                    out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)

                    # Save the mask
                    this_out_path = path.join(self.out_path, vid_name)
                    os.makedirs(this_out_path, exist_ok=True)
                    out_mask = mapper.remap_index_mask(out_mask)
                    out_img = Image.fromarray(out_mask)
                    if vid_reader.get_palette() is not None:
                        out_img.putpalette(vid_reader.get_palette())
                    out_img.save(os.path.join(this_out_path, frame[:-4]+'.png'))

        ps = subprocess.run(
            [
                "ffmpeg",
                "-framerate","24",
                "-pattern_type","glob","-i","./results/default_video/*.png", 
                "-c:v","libx264",
                "-crf","0",
                "-y",
                self.outvideo_path
             ],
        )
        
        return Path(self.outvideo_path)