**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[german1]: ./report_materials/german1.png ""
[german2]: ./report_materials/german2.png ""
[german3]: ./report_materials/german3.png ""
[german4]: ./report_materials/german4.png ""
[german5]: ./report_materials/german5.png ""
[test_german1]: ./report_materials/test_german1.png ""
[test_german2]: ./report_materials/test_german2.png ""
[test_german3]: ./report_materials/test_german3.png ""
[test_german4]: ./report_materials/test_german4.png ""
[test_german5]: ./report_materials/test_german5.png ""
[stats]: ./report_materials/stats.png ""
[stats_org]: ./report_materials/stats.png ""
[norm]: ./report_materials/norm.png ""
[samples]: ./report_materials/samples.png ""

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 39239
* Number of testing examples = 12630
* Image data shape = (32, 32)
* Number of classes = 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. First, I visualized random samples from the dataset along with their short description and checked if they are labeled accordingly.

![alt text][samples]

Next, I used histogram visualization to understand the distribution of each class in training, validation and test dataset. In the original distribution, it is very easy to see number of samples in each class is not distributed equally. 


As this might cause a bias in the training process, I used augmentation techniques to increase the number of samples in underrepresented classes. You can see the distribution after the augmentation in the same chart below:

![alt text][stats_org]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert images into grayscale. Next, I normalized the data to descrease the effects of different illumination. 

Here is the example after grayscale conversion and normalization steps.

![alt text][norm]

As a last but not least, I decided to generate additional data because I noticed that some classes are overrepresented with a lot of number samples and some classes underrepresented with a small number of samples. I used augmentation technique which generates randomly distorted the images in the underrepresented classes until the number of samples in each class reaches average number of samples for each class in the dataset. After the augmentation step, underrespesented classes have more number of samples as can be seen in the histogram plot below:

![alt text][stats]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, 8 depth	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, 16 depth	|
| RELU					|												|
| Max pooling	      	| 2x2 stride 				|
| Convolution 3x3     	| 1x1 stride, valid padding, 32 depth	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, 32 depth	|
| RELU					|												|
| Max pooling	      	| 2x2 stride				|
| Convolution 3x3     	| 1x1 stride, valid padding, 32 depth	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Fully connected		| 288x128        									|
| RELU					|												|
| Dropout					|												|
| Fully connected		| 128x84        									|
| RELU					|												|
| Dropout					|												|
| Fully connected		| 84x43(nr of classes)        									|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer with initial learning rate of 0.0001. I used mean of softmax cross entropy as a loss function. I used different numbers of epoch but realized that my model continues to learn at each epoch and kept epoch number 20. Here is the validation accuracy I got for each epoch in the training process.

```
EPOCH 1 ...
Validation Accuracy = 0.835

EPOCH 2 ...
Validation Accuracy = 0.923

EPOCH 3 ...
Validation Accuracy = 0.947

EPOCH 4 ...
Validation Accuracy = 0.960

EPOCH 5 ...
Validation Accuracy = 0.975

EPOCH 6 ...
Validation Accuracy = 0.975

EPOCH 7 ...
Validation Accuracy = 0.972

EPOCH 8 ...
Validation Accuracy = 0.979

EPOCH 9 ...
Validation Accuracy = 0.974

EPOCH 10 ...
Validation Accuracy = 0.973

EPOCH 11 ...
Validation Accuracy = 0.978

EPOCH 12 ...
Validation Accuracy = 0.977

EPOCH 13 ...
Validation Accuracy = 0.976

EPOCH 14 ...
Validation Accuracy = 0.978

EPOCH 15 ...
Validation Accuracy = 0.978

EPOCH 16 ...
Validation Accuracy = 0.983

EPOCH 17 ...
Validation Accuracy = 0.980

EPOCH 18 ...
Validation Accuracy = 0.980

EPOCH 19 ...
Validation Accuracy = 0.984

EPOCH 20 ...
Validation Accuracy = 0.980
```

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

I started the project with a simple network model and without data augmentation techniques. At that time, I got around 0.85 percent validation accuracy which is not very satisfactory. Before playing with the network architecture, I wanted to use some preprocessing and augmentation techniques because I know these steps are mandatory for each network architecture. Also, I feel changing the network structure requires more expertise and it might be a time consuming to change it even if one has some intuition about the structure. 

As a first step, I added normalization step. Simple way to normalize the data is to divide by 128 and substract 1. You'll always get numbers between -1 and 1. However, more sophisticated way is to find the mean and standard deviation of the input and substact and divide by mean and standard deviation, respectively. By this way, I got better validation accuracy.

Next important step was to augmentation of the dataset which helped much to get great 0.965 test accuracy. I augmented underrepresented classes with the randomly distorted images from the same class.

As a last step, I played with the network. Instead of constructing network from scratch, I used already known network used for traffic sign recognition. This network performed relatively well compared to the simple LeNet network that I used previously. 

My final model results were:
* validation set accuracy of 0.98
* test set accuracy of 0.965

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][german1] 
![alt text][german2] 
![alt text][german3] 
![alt text][german4] 
![alt text][german5]

The first image might be difficult to classify because I changed the aspect ratio a little but much. Also, dataset does not contain representative class for the 3rd image and I wanted to test if the network will recognize it as another class or it will make predictions with low confidence.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][test_german1] 
![alt text][test_german2] 
![alt text][test_german3] 
![alt text][test_german4] 
![alt text][test_german5]


The model was able to correctly guess 3 of the 5 traffic signs with a very high confident. 4th traffic sign is also predicted correctly but with a relatively low confidence. 3rd sign is predicted as speed limit with 0.87 confidence altough this sign class was not in the dataset. Still, the overall template of this sign is very similar to speed limit sign however I would expect lower confidence in a more sophisticated recognition network. 


