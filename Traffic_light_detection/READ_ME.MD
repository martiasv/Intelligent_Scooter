
The idea:

To apply more modern approaches like Faster-CNN, YOLO, and SSD on datasets of trafficlights and compare their performance and computational cost. The goal is to later shrink this into lighter versions running on a Raspberry Pi.

To combat noise, use adaptive correlation filters (?).

Thoughts on different approaches:

1. **Using a pretrained network to detect the ROI of the traffic lights, then cropping them out and:**
    a. **Using classic computer vision methods to distinguish the colors**
    Strength: These can be very fast compared to their deep learning counterparts.
    Weaknesses: detecting shapes and colors can be very difficult. Red and yellow light can be very similar in color given the lighting in the surroundings. Modern traffic lights are comprised of small LEDs, so when the bulbs go out, the form of the light itself is not circular anymore. And not to mention that the traffic lights in Barcelona are yellow themselves.
    b. **Using another network to detect the states**
    Strength: can be designed to be robust to changes in lighting.
    Weaknesses: requires designing and training a separate network which will take up additional computational resources when running.

2. **Using a network to detect both the traffic lights and their states**
Strength: both detection and classification of traffic lights can be done at the same time in a robust manner. 
Weaknesses: black-box approach to classification of traffic lights. 
