VapourSynth image segmentation with [OpenCV Hierarchical Feature Selection module](https://github.com/opencv/opencv_contrib/tree/4.x/modules/hfs).\
PS: CUDA GPU is a requirement.

### Usage
```python
cv_hfs.HFS(vnode clip[, float segEgbThresholdI=0.08, int minRegionSizeI=100, float segEgbThresholdII=0.28, int minRegionSizeII=200, float spatialWeight=0.6, int slicSpixelSize=8, int numSlicIter=5])
```
### Parameters:

- clip\
    A clip to process. RGB24 format only.
- segEgbThresholdI\
    It is a constant used to threshold weights of the edge when merging adjacent nodes when applying EGB algorithm.\
    The segmentation result tends to have more regions remained if this value is large and vice versa.
- minRegionSizeI\
    After the EGB segmentation, regions that have fewer pixels then this parameter will be merged into it's adjacent region.
- segEgbThresholdII\
    It serves the same purpose as segEgbThresholdI. The segmentation result tends to have more regions remained if this value is large and vice versa.
- minRegionSizeII\
    It serves the same purpose as minRegionSizeI.
- spatialWeight\
    It describes how important is the role of position when calculating the distance between each pixel and it's center.\
    The exact formula to calculate the distance is colorDistance+spatialWeight×spatialDistance.\
    The segmentation result tends to have more local consistency if this value is larger.
- slicSpixelSize\
    It describes the size of each superpixel when initializing SLIC.\
    Every superpixel approximately has slicSpixelSize×slicSpixelSize pixels in the beginning.
- numSlicIter\
    It describes how many iteration to perform when executing SLIC.
