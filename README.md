# IPN-Hand dataset 
set up dataset for tensorflow training

Files:
- `data_test.py`: example usage
- `data_utils.py`: some utility functions `@ianzur` used to check if things work

Notes:
- requires manual download of [IPN-Hand data](https://gibranbenitez.github.io/IPN_Hand/) 
  > only "frames" and "annotations" folders
- Not all video files contain the same number of frames as labeled in the annotation or compared to # of images in `frames/<video_name>` folders. see: [IPN-hand issue #11](https://github.com/GibranBenitez/IPN-hand/issues/11)
