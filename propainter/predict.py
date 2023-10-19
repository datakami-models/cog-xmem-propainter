# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from typing import List

from cog import BasePredictor, Input, Path
import propainter.inference_propainter as propainter


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device, self.fix_raft, self.fix_flow_complete, self.model = propainter.setup()

    def predict(
        self,
        video: Path = Input(description="Input video"),
        mask: Path = Input(description="Mask for video inpainting. Can be a static image (jpg, png) or a video (avi, mp4).", default=None),
        return_input_video: bool = Input(description="Return the input video in the output.", default=False),
        resize_ratio: float = Input(description="Resize scale for processing video.", default=1.0),
        height: int = Input(description="Height of the processing video.", default=-1),
        width: int = Input(description="Width of the processing video.", default=-1),
        mask_dilation: int = Input(description="Mask dilation for video and flow masking.", default=4),
        ref_stride: int = Input(description="Stride of global reference frames.", default=10),
        neighbor_length: int = Input(description="Length of local neighboring frames.", default=10),
        subvideo_length: int = Input(description="Length of sub-video for long video inference.", default=80),
        raft_iter: int = Input(description="Iterations for RAFT inference.", default=20),
        mode: str = Input(description="Modes: video inpainting / video outpainting. If you want to do video inpainting, you need a mask. For video outpainting, you need to set scale_h and scale_w, and mask is ignored.", choices=['video_inpainting', 'video_outpainting'], default='video_inpainting'),
        scale_h: float = Input(description="Outpainting scale of height for video_outpainting mode.", default=1.0),
        scale_w: float = Input(description="Outpainting scale of width for video_outpainting mode.", default=1.0),
        save_fps: int = Input(description="Frames per second.", default=24),
        fp16: bool = Input(description="Use fp16 (half precision) during inference. Default: fp32 (single precision).", default=False)
        
    ) -> List[Path]:
        """Run a single prediction on the model"""
        output = "results"
        save_frames=False

        if mode in ['video_inpainting']:
            if not mask:
                raise ValueError("Video inpainting needs a mask input.")
            if mask and mask.suffix.lower() not in ['.mp4','.avi','.png','.jpg']:
                raise ValueError("ProPainter via cog only supports static masks as .jpg or .png, or video masks as .avi and .mp4.")

        if mode in ['video_outpainting']:
            if not scale_w or not scale_h:
                raise ValueError("Video outpainting needs scale_h and scale_w as input parameters.")
        
        in_video, out_video = propainter.infer(self.device, self.fix_raft, self.fix_flow_complete, self.model, str(video), str(mask), output, resize_ratio, height, width, mask_dilation, ref_stride, neighbor_length, subvideo_length, raft_iter, mode, scale_h, scale_w, save_fps, save_frames, fp16)
        if return_input_video:
            return [Path(in_video), Path(out_video)]
        return [Path(out_video)]
