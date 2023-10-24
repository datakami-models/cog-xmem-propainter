# cog-xmem-propainter
Cog pipeline for XMem and ProPainter

[![Replicate](https://replicate.com/jd7h/xmem-propainter-pipeline/badge)](https://replicate.com/jd7h/xmem-propainter-inpainting/)

This is a [cog](https://github.com/replicate/cog) image for a generative AI pipeline that combines two models: 
- [XMem](), a model for video object segmentation
- [ProPainter](), a model for video inpainting

This pipeline that can be used for easy video inpainting. 
XMem turns a source video and an annotated first frame into a video mask.
ProPainter takes a source video and a video mask and fills everything under the mask with inpainting. 

## How to use it
XMem needs an annotated first video frame to create a video mask for ProPainter.
To make this annotated frame, you can extract the frames from your source video with ffmpeg:

```
ffmpeg -i kitten_short.mp4 frames/%04d.jpg
```

You can then use an image segmentation model, such as [Segment Anything](https://replicate.com/yyjim/segment-anything-everything), to turn the first frame, `frames/0001.jpg`, into a mask.

We can now feed our video `kitten_short.mp4` and `first_frame_mask.png` into this pipeline. 
XMem will generate a video mask from the inputs. ProPainter will take XMem's output, and use it for video inpainting.

You can see [an example on Replicate](). This repository also contains two examples in `example`, that you can use to test the pipeline.

## License
The cog files have the MIT license. For the license of the used models, please see their respective repositories on Github.
- [XMem license](https://github.com/jd7h/XMem/blob/main/LICENSE)
- [ProPainter license](https://github.com/jd7h/ProPainter/blob/main/LICENSE)
