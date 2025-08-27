# Filter Cleaning Runs
This tool filters recorded videos from street cleaners and extracts only these sections (for further processing and model training)

## How to use
Samples:
```sh
# Simplest call
python filter_cleaning_runs.py ./input ./output

# Filtering out waste facility area
python filter_cleaning_runs.py ./input ./output --facility-area 52.42691465669094,10.855876062481274,52.41970634676204,10.875833665085436
``` 
The script has many configuration parameters for segment filtering. The defaults are sensible, but please check the scripts help output for more information.

## How it works
The gps companion files (same file name prefix as video files) are analyzed for cleaning runs by calculating the speed (using a sliding average with a window of a configurable size over GPS positions) and then determining the sections where the street cleaner is slower than a certain threshold (configurable as a CLI parameter) for longer than a certain time (to rule out waiting at a red light, being stuck in slow traffic, etc.; also configurable through CLI).
Also, in order to filter out movements at the waste facility itself a rectangular area can be optionally defined by adding two pairs of geo coordinates (the top-left and lower-right corner).
In an (optional) second step a classifier is run that determines cleaning status (i.e. if the brushes are engaged) visually. As this is a slowly changing property we only need to look at frames in a certain stride (default 1s, can be configured).

## Input data
- A directory of files which always exist in pairs (the recordings start at the exact same time)
  - A video file with a file name similar to `2025-07-17_14-10-32_video.mkv` (the prefix is a timestamp)
  - A GPS log file containing timestamped NMEA messages with a file name similar to `2025-07-17_14-10-32_gps.log`
    - Sample line (GGA messages are of interest): `2025-07-17T14:11:13,772495840+02:00;$GPGGA,061233.00,5224.134571,N,01044.999688,E,1,04,1.6,96.6,M,46.0,M,,*5B`