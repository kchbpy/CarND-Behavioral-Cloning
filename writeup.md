**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./pics/architecture.jpg "Model Visualization"
[image2]: ./pics/1.jpg "Clock-wise"
[image3]: ./pics/2.jpg "Counter-clockwise"
[image4]: ./pics/3.jpg "Recovery Image"
[image5]: ./pics/4.jpg "Recovery Image"
[image6]: ./pics/5.jpg "Recovery Image"
[image7]: ./pics/6.jpg "Recovery Image"
[image8]: ./pics/7.jpg "Recovery Image"
[image9]: ./pics/8.jpg "Recovery Image"
[image10]: ./pics/9.jpg "Recovery Image"
[image11]: ./pics/10.jpg "Recovery Image"
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* video.mp4 is the recording of my vehicle driving autonomously
#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable 

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is refer to the NVIDIA model(code line 68-79).The kernel size is differnt between my model and NVIDIA model. 

To reduce the training time, I shrinked the images(code line 30,35,39).

The model includes RELU layers to introduce nonlinearity.

I didn't use the lambda layer because when I use that layer I will get an error.Instead, I normalized the data after reading the data(code line 44).

#### 2. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 81).

#### 4. Appropriate training data

I used the the data provided by Udacity and the data I created.

I also used the right and left views.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Final Model Architecture

The final model architecture (model.py lines 68-79) consisted of five convolution layers and three full connection layers.   

Here is a visualization of the architecture 

![alt text][image1]

#### 2. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded the vehicle track in a clock-wise direction. Here is an example image of clock-wise driving:

![alt text][image2]



I then recorded the track in a 
counter-clockwise direction to combat the bias. Here is an example image of counter-clockwise driving:

![alt text][image3]

Then I recorded recorvering driving. Then I picked off the data whoes angle is 0 in this recorvering lap. 

![alt text][image4]

![alt text][image5]

![alt text][image6]

![alt text][image7]

![alt text][image8]

![alt text][image9]

![alt text][image10]

![alt text][image11]


After the collection process, I had 32502 number of data points.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.
