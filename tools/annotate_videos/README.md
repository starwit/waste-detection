# Filter visible trash
This tool reads video files from a directory and inferences all images using Ultralytics YOLO detection mode with a given weights file. It then saves only images that exceed a certain object count (of a certain class) to its output directory.
In order to get the a high dataset variety, a minimum frame interval can be specified (i.e. how many frames two picked frames need to be apart).

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