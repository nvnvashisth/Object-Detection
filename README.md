# Object-Detection


# Task 1 Image processing and HOG descriptor

OpenCV is a famous and popular open source Computer Vision library. It is released under
a BSD license and hence it’s free for both academic and commercial use. In this exercise,
OpenCV for C++ is required for all the implementation. In this task, you’re required to
master the basic usage of OpenCV on image processing and then use the built-in class
HOGDescriptor to detect HOG descriptors on given images.

![alt text](https://github.com/nvnvashisth/Object-Detection/blob/master/Output/hog_vis.jpg)

# Task 2 Object classification

After extraction of HOG descriptors from a image, we can use it to train a classifier. In this
task, we will use Binary Decision Tree and Random Forest to classify images using their
HOG descriptors. A Random Forest is an ensemble of Random Trees. By aggregating all
the predictions from different trees, a forest can in general yield a more robust prediction
than a single tree. The relationship between a Random Tree and a Random Forest

For the implementation of Random Tree, we use the Binary Decision Tree provided by
OpenCV. It is defined as cv::ml::DTrees8

• Create a decision tree – cv::ml::DTrees.create()
• Some parameters to set:
– void setCVFolds( int val ); // set num cross validation folds
– void setMaxCategories( int val ); // set max number of categories
– void setMaxDepth( int val ); // set max tree depth
– void setMinSampleCount( int val ); // set min sample count

• Train a decision tree – cv::ml::DTrees.train()
• Predict class using decision tree – cv::ml::DTrees.predict()
After being able to do classification with one Binary Decision Tree, you’re required to
implement a Random Forest class composed of a group of Binary Decision Trees. You have
to implement at least those three methods:
• create – construct a forest with a given number of trees and initialize all the trees with
given parameters
• train – train each tree with a random subset of the training data
• predict – aggregate predictions from all the trees and vote for the best classification
result as well as the confidence (percentage of votes for that winner class)




# Task 3 Object detection

In this task, you will need to detect objects in images with random forest.
In the training stage, you have images from different objects organized in different
directories. Especially, there’s one background class, whose images are generated from the
possible backgrounds you would see in the test images. You should train your Random
Forest with the capability to distinguish between those images from different classes. Note
that the data you have for training may not be sufficient enough, you may need to
augment it (add rotation, flip etc.) to generate more samples for training. 

![alt text](https://github.com/nvnvashisth/Object-Detection/blob/master/Output/image.jpg)
![alt text](https://github.com/nvnvashisth/Object-Detection/blob/master/Output/image_clone_1.jpg)
![alt text](https://github.com/nvnvashisth/Object-Detection/blob/master/Output/image_clone_2.jpg)
![alt text](https://github.com/nvnvashisth/Object-Detection/blob/master/Output/image_clone_3.jpg)
![alt text](https://github.com/nvnvashisth/Object-Detection/blob/master/Output/image_clone_4.jpg)
![alt text](https://github.com/nvnvashisth/Object-Detection/blob/master/Output/image_test.jpg)

