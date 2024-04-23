## YVR Hackathon Machine Vision

For the 2024 YVR Hackathon Crow's nest Facility Detection System. Machine learning model that utilizes YOLOv8n to inference live feed cams. Model was tuned to work better at an airport setting to detect passengers, staff, baggage, unattendent baggage and garbage. The program will then send an alert via email to a designated email with a description of what the camera has captured in the past few seconds.
 
## Challenges
1. Identifying areas that require attention from staff due to high traffic volume and observed waste
2. Detect unattended baggage, garbage, laptops, etc.

*Note: Necessary dependencies to run
1. `pip install ultralytics supervision opencv-python numpy`

`downloader.py` downloads an annotated video after inferencing with our model
`cam_detect.py` runs the streaming from the camera and inference

  