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
        # note: when we call Predictor.predict() from outside cog, the default values for arguments are not passed to the method.
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
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)


# self,
#         video: Path = Input(description="Input video"),
#         mask: Path = Input(description="Mask for video inpainting. Can be a static image (jpg, png) or a video (avi, mp4).", default=None),
#         return_input_video: bool = Input(description="Return the input video in the output.", default=False),
#         resize_ratio: float = Input(description="Resize scale for processing video.", default=1.0),
#         height: int = Input(description="Height of the processing video.", default=-1),
#         width: int = Input(description="Width of the processing video.", default=-1),
#         mask_dilation: int = Input(description="Mask dilation for video and flow masking.", default=4),
#         ref_stride: int = Input(description="Stride of global reference frames.", default=10),
#         neighbor_length: int = Input(description="Length of local neighboring frames.", default=10),
#         subvideo_length: int = Input(description="Length of sub-video for long video inference.", default=80),
#         raft_iter: int = Input(description="Iterations for RAFT inference.", default=20),
#         mode: str = Input(description="Modes: video inpainting / video outpainting. If you want to do video inpainting, you need a mask. For video outpainting, you need to set scale_h and scale_w, and mask is ignored.", choices=['video_inpainting', 'video_outpainting'], default='video_inpainting'),
#         scale_h: float = Input(description="Outpainting scale of height for video_outpainting mode.", default=1.0),
#         scale_w: float = Input(description="Outpainting scale of width for video_outpainting mode.", default=1.0),
#         save_fps: int = Input(description="Frames per second.", default=24),
#         fp16: bool = Input(description="Use fp16 (half precision) during inference. Default: fp32 (single precision).", default=False)

