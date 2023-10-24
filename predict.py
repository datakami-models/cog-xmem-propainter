# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
from typing import List

from cog import BasePredictor, Input, Path

import propainter.predict as propainter
import xmem.predict as xmem


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print("Setting up XMem...")
        self.xmem_predictor = xmem.Predictor()
        self.xmem_predictor.setup()
        print("Setting up ProPainter...")
        self.propainter_predictor = propainter.Predictor()
        self.propainter_predictor.setup()

    def predict(
        self,
        video: Path = Input(description="Source video for object segmentation"),
        mask: Path = Input(description="Segmentation mask for the first frame of the video, of the object(s) we want to segment."),
        mask_dilation: int = Input(description="Extra border around the mask in pixels", default=4),
        return_intermediate_outputs: bool = Input(description="Return the intermediate processing results in the output.", default=True),
        fp16: bool = Input(description="Use half-precision (fp16), instead of full precision. This speeds up results.", default=True),
    ) -> List[Path]:
        
        # note: when we call Predictor.predict() from outside cog, 
        # the default values for arguments are not passed to the method.
        xmem_output = self.xmem_predictor.predict(video=video, mask=mask)
        propainter_outputs = self.propainter_predictor.predict(
            video=video, 
            mask=xmem_output, 
            return_input_video=return_intermediate_outputs,
            resize_ratio=1.0,
            height=-1,
            width=-1,
            mask_dilation=mask_dilation,
            ref_stride=10,
            neighbor_length=10,
            subvideo_length=80,
            raft_iter=20,
            mode="video_inpainting",
            scale_h=1.0,
            scale_w=1.0,
            save_fps=24,
            fp16=fp16,
        )
        
        if return_intermediate_outputs:
            return [xmem_output] + propainter_outputs
        return propainter_outputs

