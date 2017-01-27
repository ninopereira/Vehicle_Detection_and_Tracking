
# coding: utf-8

# In[1]:

# import methods

from methods import plot3d
from methods import bin_spatial
from methods import color_hist
from methods import get_hog_features
from methods import extract_features
from methods import slide_window

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
# NOTE: the next import is only valid 
# for scikit-learn version <= 0.17
# if you are using scikit-learn >= 0.18 then use this:
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split


# # Method
# * Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
# * Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
# * Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
# * Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
# * Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
# * Estimate a bounding box for vehicles detected.

# # Combined Color, Histogram and HOG Classification

# In[2]:

# Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier

cars = []
notcars = []

# load vehicle images
images = glob.iglob('vehicles/**/*.png', recursive=True)
for image in images:
   cars.append(image)
   
# load non vehicle images
images = glob.iglob('non-vehicles/**/*.png', recursive=True)
for image in images:
   notcars.append(image)

print('cars = ',len(cars))
print('notcars = ',len(notcars))

orient = 9
pix_per_cell = 8
cell_per_block = 2

car_features = extract_features(cars, cspace='RGB', spatial_size=(32, 32),
                       hist_bins=32, hist_range=(0, 256))
notcar_features = extract_features(notcars, cspace='RGB', spatial_size=(32, 32),
                       hist_bins=32, hist_range=(0, 256))


# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X - normalise data
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
   scaled_X, y, test_size=0.2, random_state=rand_state)

# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(t2-t, 'Seconds to train SVC...')
# Check the score of the SVC
print('Train Accuracy of SVC = ', svc.score(X_train, y_train))
print('Test Accuracy of SVC = ', svc.score(X_test, y_test))
# Check the prediction time for a single sample
t=time.time()
prediction = svc.predict(X_test[0].reshape(1, -1))
print("prediction",prediction)
t2 = time.time()
print(t2-t, 'Seconds to predict with SVC')


# # Sliding Window Implementation

# In[3]:

def draw_rectangles(img,window_list,color= (255,255,255)):
    labeled_img = img.copy()
    for window in window_list:
        pt1 = window[0]
        pt2 = window[1]
        thickness = 4
        cv2.rectangle(labeled_img, pt1, pt2, color, thickness)
    return labeled_img


# In[4]:

# Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
img = cv2.imread('test_images/test1.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
height, width, channels = img.shape
print('height, width, channels = ',height, width, channels)

window_list=();

step_h = 32
start_h = step_h#int(height/4)
stop_h = height 
size_of_sq = int(256 * (1/height))
original_y_val = int(9*height/16) #int(6*height/11)
# original_y_val = int(6*height/12)
y_val = original_y_val
overlap = 0.75
rectangles = []
while y_val<height:
    size_of_sq = int(500000 * (y_val/height) ** 16)
#     size_of_sq = int(3000 * (y_val/height) ** 8)
    window_list = slide_window(img, x_start_stop=[0, width], y_start_stop=[y_val,y_val+size_of_sq], 
                    xy_window=(size_of_sq, size_of_sq), xy_overlap=(overlap,overlap))
    rectangles.extend(window_list)

    y_val = y_val + (1-overlap)*size_of_sq

print("num rectangles = ", len(rectangles))
labeled_img = draw_rectangles(img,rectangles)
plt.imshow(labeled_img)
plt.show()
plt.imshow(img)
plt.show()


# In[6]:

detected_img = img.copy()
for rectangle in rectangles:
    pt1 = rectangle[0]
    pt2 = rectangle[1]
    crop_img = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
    size = (64,64)
    crop_img = cv2.resize(crop_img, size)

    img_features =[]
    hog_channel=0
    spatial_features = bin_spatial(crop_img, size=(32,32))
    hist_features = color_hist(crop_img, nbins=32, bins_range=(0,256))
    hog_features = get_hog_features(crop_img[:,:,hog_channel], orient=9, 
                    pix_per_cell=8, cell_per_block=2, vis=False, feature_vec=True)
    img_features.append(np.concatenate((spatial_features, hist_features, hog_features)))
    X = np.vstack((img_features)).astype(np.float64)                        
    X = X.reshape(1, -1)
    scaled_X = X_scaler.transform(X)
    prediction = svc.predict(scaled_X.reshape(1, -1))
    if prediction == 1:
        thickness = 4
        cv2.rectangle(detected_img, pt1, pt2, (255,255,255),thickness)

plt.imshow(detected_img)
plt.show()

