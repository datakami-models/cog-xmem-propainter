# cog-xmem-propainter
Cog pipeline for XMem and ProPainter

[![Replicate](https://replicate.com/jd7h/xmem-propainter-pipeline/badge)](https://replicate.com/jd7h/xmem-propainter-inpainting/)

This is a [cog](https://github.com/replicate/cog) image for a generative AI pipeline that combines two models: 
- [XMem](https://github.com/jd7h/XMem), a model for video object segmentation
- [ProPainter](https://github.com/jd7h/ProPainter), a model for video inpainting

This pipeline can be used for easy video inpainting. 
XMem turns a source video and an annotated first frame into a video mask.
ProPainter takes a source video and a video mask and fills everything under the mask with inpainting. 

## How to use it
Here's how you can use this pipeline to do video inpainting on a source video `kitten_short.mp4`.

### 1. Extract the first frame of your video.

XMem needs an annotated first video frame to create a video mask for ProPainter.
To make this annotated frame, you can extract the frames from your source video with ffmpeg:

```
ffmpeg -i kitten_short.mp4 frames/%04d.jpg
```

### 2. Create a mask of the first frame of your video
You can then use an image segmentation model, such as [Segment Anything](https://replicate.com/yyjim/segment-anything-everything), to turn the first frame, `frames/0001.jpg`, into a mask.

### 3. Feed the source video and the mask into the pipeline
We can now feed our video `kitten_short.mp4` and `first_frame_mask.png` into this pipeline. 
XMem will generate a video mask from the inputs. ProPainter will take XMem's output, and use it for video inpainting.

#### On Replicate
You can run this model directly [on Replicate](https://replicate.com/jd7h/xmem-propainter-inpainting/). Upload the source video under 'video', and the mask under 'mask'. :)

#### On your computer
If you have docker and cog installed, you can run this model locally by cloning this repository, and running the following commands:
```
cog build
cog predict -i video=@example/kitten_short.mp4 -i mask=@example/kitten_masked_first_frame.png
```
You can see [example runs of this model on Replicate](https://replicate.com/jd7h/xmem-propainter-inpainting/). This repository also contains two example inputs that can be used to test the pipeline.

## License
The cog files have the MIT license. For the license of the underlying models, please see their respective repositories on Github.
- [XMem license](https://github.com/jd7h/XMem/blob/main/LICENSE)
- [ProPainter license](https://github.com/jd7h/ProPainter/blob/main/LICENSE)
