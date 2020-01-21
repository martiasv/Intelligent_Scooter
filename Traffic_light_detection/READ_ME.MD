The goal is to detect traffic lights and their state in real-time minimizing the computational cost.

Ideas:
1. feed images containing traffic lights directly to a NN which will output the location and the states of the traffic lights
2. Feed the images directly to a NN to find the location of the traffic light and use classic computer vision techniques to detect its state (shapes,contours,colors)
3. Use Haar-features + Adaboost to detect the traffic light, filter the results with a NN to obtain location. Use classic computer vision techniques to detect its state.

To combat noise, use adaptive correlation filters.