## Data Description for Task 3

1. There are 3 objects used in this task, all the images from each object are put in the same directory. 
2. The train directory contains all the cropped images for each class for training. directory train/03 stores images for background, which could make the detector better to distinguish between object and non-object in the detection stage. 
3. The test directory contains 44 test images, you have to do detection through all of them. In the gt directory, you can get all the ground truth bounding boxes for different objects in each image, the format is **class_id top_left_x top_left_y bottom_right_x bottom_right_y**.
4. The ground truths are labeled with squared bounding boxes.
