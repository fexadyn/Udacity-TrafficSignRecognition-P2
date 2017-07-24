
# Self-Driving Car Engineer Nanodegree

## Deep Learning

## Project: Build a Traffic Sign Recognition Classifier

In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 

In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.

The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.


>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.


```python
a = np.array([[[[3],[3],[3]],[[3],[3],[3]],[[3],[3],[3]]]])

print(a[:,:,:,:,np.newaxis].shape)
```

---
## Step 0: Load The Data


```python
# Load pickled data
import pickle
import collections
import numpy as np
import csv
import sys
import cv2


def rgb2gray(img):
    return (img[:,:,:,0]/3.0+img[:,:,:,1]/3.0+img[:,:,:,2]/3.0)

#applies random affine transformation to given image
def random_affine_tfm(img):
    
    image_shape = (32,32)
    
    rot = np.random.randint(-5,5)
    trans_x = np.random.randint(-3,3)
    trans_y = np.random.randint(-3,3)

    M = cv2.getRotationMatrix2D((image_shape[0]/2.0,image_shape[1]/2.0),rot,1)   
    img_warped = cv2.warpAffine(img,M,image_shape)

    M = np.float32([[1,0,trans_x],[0,1,trans_y]])
    img_warped = cv2.warpAffine(img_warped,M,image_shape)
    
    return img_warped

def augment_dataset(nr_elements, X_train_org, y_train_org):
    y_train_freq = collections.Counter(y_train_org)
    
    X_train = X_train_org.copy()
    y_train = y_train_org.copy()
        
    for key, value in y_train_freq.most_common():
        if value < nr_elements:
            all_indices = [index for index, x in enumerate(y_train_org) if x == key]
            X_train_aug = np.empty((0,32,32),int)
            y_train_aug = np.empty((0,1),int)
            
            for i in range(nr_elements-value):
                rand_indice = all_indices[np.random.randint(0,len(all_indices)-1)]
                img_warped = random_affine_tfm(X_train_org[rand_indice])

                X_train_aug = np.append(X_train_aug,[img_warped],axis=0)
                y_train_aug = np.append(y_train_aug,key)
            X_train = np.append(X_train,X_train_aug,axis=0)
            y_train = np.append(y_train, y_train_aug,axis=0)
            
    
    return X_train, y_train


# TODO: Fill this in based on where you saved the training and testing data

training_file = "traffic-signs-data/train.p"
validation_file = "traffic-signs-data/valid.p"
testing_file = "traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train_org, y_train_org = rgb2gray(train['features']), train['labels']
X_valid_org, y_valid = rgb2gray(valid['features']), valid['labels']
X_test_org, y_test = rgb2gray(test['features']), test['labels']
sign_list = {}

#X_train_aug, y_train_aug = X_train_org.copy(),y_train_org.copy()

X_train_aug, y_train_aug = augment_dataset(nr_elements=500,X_train_org=X_train_org,y_train_org=y_train_org)


with open('signnames.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        try:
            sign_list[int(row[0])] = row[1]
        except ValueError:
            pass

signs_desc = []
for sign_key, sign_value in sorted(sign_list.items()):
    print(sign_key, sign_value)
    signs_desc.append(sign_value)
    

```

    0 Speed limit (20km/h)
    1 Speed limit (30km/h)
    2 Speed limit (50km/h)
    3 Speed limit (60km/h)
    4 Speed limit (70km/h)
    5 Speed limit (80km/h)
    6 End of speed limit (80km/h)
    7 Speed limit (100km/h)
    8 Speed limit (120km/h)
    9 No passing
    10 No passing for vehicles over 3.5 metric tons
    11 Right-of-way at the next intersection
    12 Priority road
    13 Yield
    14 Stop
    15 No vehicles
    16 Vehicles over 3.5 metric tons prohibited
    17 No entry
    18 General caution
    19 Dangerous curve to the left
    20 Dangerous curve to the right
    21 Double curve
    22 Bumpy road
    23 Slippery road
    24 Road narrows on the right
    25 Road work
    26 Traffic signals
    27 Pedestrians
    28 Children crossing
    29 Bicycles crossing
    30 Beware of ice/snow
    31 Wild animals crossing
    32 End of all speed and passing limits
    33 Turn right ahead
    34 Turn left ahead
    35 Ahead only
    36 Go straight or right
    37 Go straight or left
    38 Keep right
    39 Keep left
    40 Roundabout mandatory
    41 End of no passing
    42 End of no passing by vehicles over 3.5 metric tons


---

## Step 1: Dataset Summary & Exploration

The pickled data is a dictionary with 4 key/value pairs:

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
- `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**

Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas


```python
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = X_train_aug.shape[0]

# TODO: Number of validation examples
n_validation = X_valid_org.shape[0]

# TODO: Number of testing examples.
n_test = X_test_org.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train_aug[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train_aug))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```

    Number of training examples = 39239
    Number of testing examples = 12630
    Image data shape = (32, 32)
    Number of classes = 43


### Include an exploratory visualization of the dataset

Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 

The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.

**NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?

### 1. Random picks from dataset


```python
### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
%matplotlib inline

import numpy as np

fig, axes = plt.subplots(3, 7, figsize=(20, 6),
                         subplot_kw={'xticks': [], 'yticks': []})

fig.subplots_adjust(hspace=0.3, wspace=0.05)

for ax in axes.flat:
    
    class_id = np.random.randint(0,n_train)
    
    ax.imshow(X_train_aug[class_id],cmap='gray')
    signs_sorted = sorted(sign_list.items())
    ax.set_title(str(y_train_aug[class_id]) + " " + signs_desc[y_train_aug[class_id]][:20])
    
plt.show()

```


![png](output_10_0.png)


### 2. Number of samples for each class


```python
def plot_dataset_hist():
    plt.figure(figsize=(20, 5))

    ax = plt.subplot(1, 1, 1)

    y_combined = [y_train_aug, y_valid, y_test]

    ax.hist(y_combined, n_classes, histtype='bar', label=['Train','Validation','Test'])
    ax.legend(prop={'size': 20})
    ax.set_title('The number of examples for each traffic sign classes')

    plt.show()
    
plot_dataset_hist()
```


![png](output_12_0.png)


----

## Step 2: Design and Test a Model Architecture

Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 

With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 

There are various aspects to consider when thinking about this problem:

- Neural network architecture (is the network over or underfitting?)
- Play around preprocessing techniques (normalization, rgb to grayscale, etc)
- Number of examples per label (some have more than others).
- Generate fake data.

Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

### Pre-process the Data Set (normalization, grayscale, etc.)

Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 

Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 

Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.


```python
### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

from sklearn.preprocessing import normalize
from sklearn.utils import shuffle

#Fixed mean and variance assumption
def normalize1(images):
    return (images - 128.) / 128.

#Normalize mean only
def normalize2(images):
    images_normalized = (images - images.mean(axis=(1,2),keepdims=True) )
    return images_normalized

#Normalize mean and variance
def normalize3(images):
    images_normalized = (images - images.mean(axis=(1,2),keepdims=True) ) / images.std(axis=(1,2),keepdims=True)
    return images_normalized


plt.figure(figsize=(20, 5))
ax = plt.subplot(1, 2, 1)

ax.imshow(X_train_aug[20],cmap='gray')
ax.set_title("Original image")

#normalize data
X_train = normalize3(X_train_aug)
X_valid = normalize3(X_valid_org)
X_test = normalize3(X_test_org)

#X_train = X_train_aug.copy()
#X_valid = X_valid_org.copy()
#X_test = X_test_org.copy()

ax = plt.subplot(1, 2, 2)

ax.imshow(X_train[20],cmap='gray')
ax.set_title("Normalized image")

plt.show()

X_train = X_train[:,:,:,np.newaxis]
X_valid = X_valid[:,:,:,np.newaxis]
X_test = X_test[:,:,:,np.newaxis]
```


![png](output_16_0.png)


### Model Architecture


```python
### Define your architecture here.
### Feel free to use as many code cells as needed.


import tensorflow as tf

EPOCHS = 20
BATCH_SIZE = 128


from tensorflow.contrib.layers import flatten

def signRecNet(x, keep_prob):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    conv1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 1, 8), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(8))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1   = tf.nn.relu(conv1)


    conv2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 8, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2   = tf.nn.relu(conv2)
    conv2   = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    
    conv3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 16, 32), mean = mu, stddev = sigma))
    conv3_b = tf.Variable(tf.zeros(32))
    conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b
    conv3   = tf.nn.relu(conv3)

    
    conv4_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 32, 32), mean = mu, stddev = sigma))
    conv4_b = tf.Variable(tf.zeros(32))
    conv4   = tf.nn.conv2d(conv3, conv4_W, strides=[1, 1, 1, 1], padding='VALID') + conv4_b
    conv4   = tf.nn.relu(conv4)
    conv4   = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    
    conv5_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 32, 32), mean = mu, stddev = sigma))
    conv5_b = tf.Variable(tf.zeros(32))
    conv5   = tf.nn.conv2d(conv4, conv5_W, strides=[1, 1, 1, 1], padding='VALID') + conv5_b
    conv5   = tf.nn.relu(conv5)

    
    fc0     = flatten(conv5)
    
    fc1_W   = tf.Variable(tf.truncated_normal(shape=(288, 128), mean = mu, stddev = sigma))
    fc1_b   = tf.Variable(tf.zeros(128))
    fc1     = tf.matmul(fc0, fc1_W) + fc1_b
    fc1     = tf.nn.relu(fc1)
    fc1     = tf.nn.dropout(fc1,keep_prob[2])

    fc2_W   = tf.Variable(tf.truncated_normal(shape=(128, 84), mean = mu, stddev = sigma))
    fc2_b   = tf.Variable(tf.zeros(84))
    fc2     = tf.matmul(fc1, fc2_W) + fc2_b
    fc2     = tf.nn.relu(fc2)
    fc2     = tf.nn.dropout(fc2,keep_prob[3])


    fc3_W   = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma))
    fc3_b   = tf.Variable(tf.zeros(n_classes))
    
    logits  = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits

```

### Train, Validate and Test the Model

A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.


```python
### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.


x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32, None)
one_hot_y = tf.one_hot(y, n_classes)
```


```python
rate = 0.001

logits = signRecNet(x,keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
```


```python
prediction = tf.nn.softmax(logits)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob:[1.0,1.0,1.0,1.0]})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
```


```python
from sklearn.utils import shuffle
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train_aug = shuffle(X_train, y_train_aug)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train_aug[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob:[0.5,0.5,0.5,0.5]})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")
```

    Training...
    
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
    
    Model saved



```python
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
```

    Test Accuracy = 0.965


---

## Step 3: Test a Model on New Images

To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.

You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

### Load and Output the Images


```python
### Load the images and plot them here.
### Feel free to use as many code cells as needed.

import glob
import matplotlib.image as mpimg

%matplotlib inline

test_images_srcs = glob.iglob('./test/*.*')

new_input = []

for image_src in test_images_srcs:
    
    
    image = mpimg.imread(image_src)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    plt.imshow(image, cmap='gray')
    plt.show()
    
    new_input.append(image)
    
```


![png](output_28_0.png)



![png](output_28_1.png)



![png](output_28_2.png)



![png](output_28_3.png)



![png](output_28_4.png)



```python


### Load the images and plot them here.
### Feel free to use as many code cells as needed.
import os
import matplotlib.image as mpimg
import PIL
from PIL import Image

signs = np.empty((0,32,32,3))
file_num=len(files)

plt.figure(figsize=(12,12))
image_list=[]
for i in range(0,file_num):
    img = Image.open(path + files[i])
    img_resized = img.resize((32,32), PIL.Image.ANTIALIAS)
    image_list.append(img_resized)
    img_array=np.array(img_resized.getdata()).\
                reshape(img_resized.size[0], img_resized.size[1], 3)
    signs = np.append(signs, [img_array], axis=0)
    plt.subplot(1, 6, i+1)
    plt.title(files[i])
    plt.imshow(img_resized)
    plt.axis('off')
    
print("Loaded ",file_num, "files")
plt.show()


```

### Predict the Sign Type for Each Image


```python
### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.

probs_op = tf.nn.softmax(logits)
top_k_op = tf.nn.top_k(probs_op, k=5)
top_k = []

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    new_input2 = np.array(new_input)[:,:,:,np.newaxis]
    top_k = sess.run(top_k_op, feed_dict={x: new_input2, keep_prob:[1.0,1.0,1.0,1.0]})
    
prob_values, class_labels = top_k[0], top_k[1]


for i in range(0,5):
    title=''
    for certainty, label in zip(prob_values[i], class_labels[i]):
        title += signs_desc[label] + ' ' + \
                    str(round(certainty* 100., 2)) + '%\n'
    plt.title(title)
    plt.axis('off')
    plt.imshow(new_input[i], cmap='gray')
    plt.show()
    
    
    for certainty, label in zip(prob_values[i], class_labels[i]):
        title += signs_desc[label] + ' ' + \
                    str(round(certainty* 100., 2)) + '%\n'
    
```


![png](output_31_0.png)



![png](output_31_1.png)



![png](output_31_2.png)



![png](output_31_3.png)



![png](output_31_4.png)


### Analyze Performance


```python
### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
```

### Output Top 5 Softmax Probabilities For Each Image Found on the Web

For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 

The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.

`tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.

Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tk.nn.top_k` is used to choose the three classes with the highest probability:

```
# (5, 6) array
a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
         0.12789202],
       [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
         0.15899337],
       [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
         0.23892179],
       [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
         0.16505091],
       [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
         0.09155967]])
```

Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:

```
TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
       [ 0.28086119,  0.27569815,  0.18063401],
       [ 0.26076848,  0.23892179,  0.23664738],
       [ 0.29198961,  0.26234032,  0.16505091],
       [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
       [0, 1, 4],
       [0, 5, 1],
       [1, 3, 5],
       [1, 4, 3]], dtype=int32))
```

Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.


```python
### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.
```

### Project Writeup

Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 

> **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

---

## Step 4 (Optional): Visualize the Neural Network's State with Test Images

 This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.

 Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.

For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.

<figure>
 <img src="visualize_cnn.png" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your output should look something like this (above)</p> 
 </figcaption>
</figure>
 <p></p> 



```python
### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
```
