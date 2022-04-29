# Endpoint-Detection
# IDS Cam Access
- To access the IDS camera via python the pyueye library must be installed. This can be done via the pip command.
- To use the IDS camera over the Jetson interface the relevent IDS software suite must be installed from the IDS website. For this project the 4.95 SDK version was installed.
- If the camera is not read after the installation modify the IPv4 setting to Link-Local Only under Network Connections.

# EndpointDetection
- The Detection folder contains the code to run the algorithm using CPU or GPU
- By modifying the main file the the algorithm can run to detect from live feed, image and video file.
- To access the GPU capabilities the OpenCV library must be built from scratch. This can be done by running the build_opencv_4.5.3.sh file. The steps expllained in the post https://yunusmuhammad007.medium.com/build-and-install-opencv-4-5-3-on-jetson-nano-with-cuda-opencl-opengl-and-gstreamer-enable-6dc7141be272 can be followed to build Opencv with CUDA.
- To optimize the YOLO detection model into a tensorRT the methodology adopted by https://github.com/jkjung-avt/tensorrt_demos can be followed.
- Post optimization the detection can be run through the endpoint.py file under tensor_rt_optimization folder.
- Both the endpoint.py and the IDSAccess.py files must be placed under the tensorrt_demos folder, this folder can be found after cloning the required repository from jkjung.

# PixelDensitySegmentation
- the pynb file under the PixelDensitySegmentation folder can be used to explore the segmentation process extended from the YOLOv4 detection model.
