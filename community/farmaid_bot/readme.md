# Farmaid bot 2.0
### What it is
The origina farmaid bot was made in response to the Arm Autonomous Robot Challange and was the winner in two categories, Best Use of AI and Most Fun Social Media Video.
It is a robot that can detect diseases in plants to allow early detection and disease mitigation.
The details can be found here https://www.hackster.io/teamato/farmaid-plant-disease-detection-robot-55eeb1
The original idea was inspired by one of the earliest videos on the tensorflow youtube channel here: https://www.youtube.com/watch?v=NlpS-DhayQA
The problems with the product described in the video was that it can be too time consuming for someone to actually walk the entire farm with such a phone. Additionally, it would either require a cloud connection to detect the diease, meaning an internet connection in the field is required the other option is to have a phone phone powerful enough to run Tensorflow Lite models and the model should work uniformly on every type of phone. To fix these issues, a simple solution is to have a standardized and cheap mobile computing platform that can run autonomously in the field and use tensorflow or tensorflow lite models to detect diseases.

The version described here is a complete overhaul of the original bot using Tensorflow 2.0, ROS and Arduino made in response to the #PoweredByTF 2.0 Challenge. 

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


### External code used
