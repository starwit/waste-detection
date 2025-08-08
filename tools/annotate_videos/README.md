# Annotate videos
This tool reads video files from a directory and inferences all images using Ultralytics YOLO detection mode with a given weights file. It then saves only images that exceed a certain object count (of a certain class) to its output directory.\
In order to get a high dataset variety, a minimum frame interval can be specified (i.e. how many frames two picked frames need to be apart).\
The purpose of this tool is to create a pre-annotated dataset with a simple model in order to speed-up labeling and automating raw data filtering.

## Parameters
- Input directory
- Input globbing pattern
- Output directory
- Object count threshold (min. count)
- Model weights to use
- Minimum interval (min. stride) between picked frames

## Output directory
- Frames as jpg files
- Labels in YOLO format
- A histogram showing the object count distribution