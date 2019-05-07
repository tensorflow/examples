# Farmaid bot 2.0
### What it is
The origina farmaid bot was made in response to the Arm Autonomous Robot Challange and was the winner in two categories, Best Use of AI and Most Fun Social Media Video.
It is a robot that can detect diseases in plants to allow early detection and disease mitigation.
The details can be found here https://www.hackster.io/teamato/farmaid-plant-disease-detection-robot-55eeb1
The original idea was inspired by one of the earliest videos on the tensorflow youtube channel here: [![Cassava Video](https://img.youtube.com/vi/NlpS-DhayQA/0.jpg)](https://www.youtube.com/watch?v=NlpS-DhayQA "cassava classification")<br>

The problems with the product described in the video was that it can be too time consuming for someone to actually walk the entire farm with such a phone. Additionally, it would either require a cloud connection to detect the diease, meaning an internet connection in the field is required the other option is to have a phone phone powerful enough to run Tensorflow Lite models and the model should work uniformly on every type of phone. To fix these issues, a simple solution is to have a standardized and cheap mobile computing platform that can run autonomously in the field and use tensorflow or tensorflow lite models to detect diseases.

The version described here is a complete overhaul of the original bot using Tensorflow 2.0, ROS and Arduino made in response to the <em>#PoweredByTF 2.0 Challenge</em>. 

### Main Components
<ol>
<li> 
While the original version was based on a Raspberry Pi, for this one we are using an Nvidia Jetson TX2 which allows us to do object detection instead of classification. Additionally, it allows us the flxeibility to add more functionality that we have planned for future expansion of the project.
We also plan a version with the Jetson Nano but that wasn't released when we started the project.
</li>
<li>
For the same reason, we have moved from a self-made solution for driving the bot to a ROS based solution what allows us to experiment with different navigation techniques, and adding additional hardware while minimizing iteration time to write custom code from scratch for each new piece of hardware.
</li>
<li>
Object Detection, for the diease detection we moved from classification in the original to object detection. This allows us to be more accurate in our results as we can tell exactly how much of a crop has been infacted and we can threshold the notifications to only trigger at particular percentages of disease for each plant/crop.
</li>
</ol>

### How it works
#### object detection
I modified some code from https://github.com/zzh8829/yolov3-tf2 (wihch is under MIT license) and added it to my fork of the repo here: https://github.com/arifsohaib/yolov3-tf2
This code is a port of YOLO object detection. The reason to use Yolo instead of the Tensorflow Object  Detection API is that the Object Detection API uses tensorflow slim which is deprecated in tensorflow 2.0<br>
We collected our own dataset from videos at the WTMC greenhouse in Ann Arbor and labeled the data using Microsoft VOTT tool. <br>
The labeled data was then exported to tfrecord format to train the model.<br>
Currently the code can only use 80 classes and needs to be fixed to take a variable number of classes from the *.names file. CHanging this by adding a num_classes flag caused other issues with the size of the output <br>
It also has an issue with training on multiple tfrecord files where it always returns nan values after 1 or 2 iterations <br>
The original repo also had some issues with the video output which I have fixed<br>
Dispite the above, when using a single tfrecord file for training, and without modifying anything else, the retraining instructions from the original repo works and the results can be seen in our video <br>
![Labeling the objects](https://i.imgur.com/KOCFGMb.jpg)
*Labeling the diseases using VOTT*

![Video Output screen](https://i.imgur.com/CZhT82E.jpg)
*Video output of detection*
#### robot software
The robotic platform is run using Robot Operating System to communicate between components. The Arudino code for the bot was written by Victor Yu and can be found here: https://github.com/victory118/farmaid_bot/tree/master/robot_launch <br>
This code contains ROS launch files to launch the various components used by the robot.<br>
The main required node is the tele_op node which can be used to operate the robot.<br>
Another important node is the usb_camera node which publishes camera output.<br>
The fiducal node which detects Aruco markers that allow the robot to build a map. Additionally, it can allow the robot to navigate autonomusly without training a neural network which is feature we plan to add later.<br>
This map will later be combined with a node that publishes the output of the tensorflow detections and the data can be combined to get a clear picture of diseases in the farm or greenhouse environment.<br>
We are also considering point cloud based navigation powered by the Intel RealSense camera.<br>

### Videos
[![Farmaid Rev2](https://img.youtube.com/vi/NipK8ffm_v0/0.jpg)](https://www.youtube.com/watch?v=NipK8ffm_v0)

### Bill of Materials
https://docs.google.com/spreadsheets/d/1RrwixLaVbXh0Zh_sp3xbb5Equx7meZaMBgM6UH-9fX0/edit?usp=sharing

### Issues + To Do List
The list is ordered in terms of importance
<ul>
<li>
The object detection code has been made to work only with the COCO dataset with 80 classes. I am currently rewriting it to work with less classes.
</li>
<li>
Gather and train on more data
</li>
<li>
We need to create a ros node to publish the object detection results
</li>
<li>
The training showed some "nan" loss values when training on several tfrecord files which need to be debugged.
</li>
<li>
We need to add autonomous navigation while being careful not to damage the operating environment.
</li>
<li>
Greenhouses tend to get warm quickly in the summer, we need to add a solution to coll the platform and the jetson.
</li>
<li>
We plan to add additional functionality like weeding, watering, and harvesting
</li>
</ul>