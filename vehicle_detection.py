
# coding: utf-8

# In[157]:

# import methods

from methods import plot3d
from methods import bin_spatial
from methods import color_hist
from methods import get_hog_features
# from methods import extract_features
from methods import slide_window

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

# NOTE: the next import is only valid 
# for scikit-learn version <= 0.17
# if you are using scikit-learn >= 0.18 then use this:
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split


# In[228]:

## General Settings

# settings for colorspace feature extraction
cspace_val = 'RGB'
spatial_size_val = (32, 32)

# settings for histogram feature extraction
hist_bins_val = 64
hist_range_val = (0,256)

# settings for hog feature extraction
orient_val = 12
pix_per_cell_val = 16
cell_per_block_val = 4
hog_channel_val = 0


# In[229]:

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
# this function combines color, histogram and hog features extraction
def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256), orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        #image = mpimg.imread(file)
        image = cv2.imread(file) # reads a file into bgr values 0-255
        
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        else: 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert to rgb
            feature_image = np.copy(image)      
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        # Call get_hog_features() with vis=False, feature_vec=True
        hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
#         features.append(hog_features)
        features.append(np.concatenate((spatial_features, hist_features, hog_features)))
#         features.append(np.concatenate((hist_features, hog_features)))    
        
    # Return list of feature vectors
    return features


# # Method
# * Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
# * Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
# * Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
# * Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
# * Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
# * Estimate a bounding box for vehicles detected.

# # Combined Color, Histogram and HOG Classification

# In[230]:

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

# experiment other color spaces like LUV, HLS

car_features = extract_features(cars, cspace=cspace_val, spatial_size=spatial_size_val,
                       hist_bins=hist_bins_val, hist_range=hist_range_val, orient=orient_val, 
                       pix_per_cell=pix_per_cell_val, cell_per_block=cell_per_block_val, hog_channel=hog_channel_val)

print("Car features extracted")

notcar_features = extract_features(notcars, cspace=cspace_val, spatial_size=spatial_size_val,
                       hist_bins=hist_bins_val, hist_range=hist_range_val, orient=orient_val, 
                       pix_per_cell=pix_per_cell_val, cell_per_block=cell_per_block_val, hog_channel=hog_channel_val)

print("Other features extracted")


# In[231]:

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
print("X_scaler ready")


# In[232]:

#save the model
joblib.dump(X_scaler, 'X_scaler_model.pkl')


# In[233]:

# Apply the scaler to X - normalise data
scaled_X = X_scaler.transform(X)
print("Data normalised")


# In[234]:

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

# RGB test accuracy = 0.9788 and low number of false positives in test images
# HSV test accuracy = 0.9777 but lots of false positives
# HLS test accuracy = 0.9688 but lots of false positives
# LUV test accuracy = 0.984  but lots of false positives
# YUV test accuracy = 0.9786 but lots of false positives  


# In[235]:

print(svc)


# In[236]:

# save the model
joblib.dump(svc, 'svc_model.pkl')


# In[237]:

# load the model
svc = joblib.load('svc_model.pkl')
X_scaler = joblib.load('X_scaler_model.pkl')
print("Model loaded: \n\n",svc)


# # Sliding Window Implementation

# In[238]:

def draw_rectangles(img,window_list,color= (255,255,255)):
    labeled_img = img.copy()
    for window in window_list:
        pt1 = window[0]
        pt2 = window[1]
        thickness = 4
        cv2.rectangle(labeled_img, pt1, pt2, color, thickness)
    return labeled_img


# In[239]:

# Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
img = cv2.imread('test_images/test1.jpg')
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
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
overlap = 0.5
rectangles = []

size_vec = [64, 96, 128, 160]
overlap_vec = [0, 0.5, 0.65, 0.8]
# size_vec = [160]
# overlap_vec = [0.8]
for i in range(len(size_vec)):
    size = size_vec[i]
    overlap = overlap_vec[i]
    window_list = slide_window(img, x_start_stop=[0, width+size], y_start_stop=[y_val,y_val+4*size], 
                    xy_window=(size, size), xy_overlap=(overlap,overlap))
    rectangles.extend(window_list)
    
# while y_val<height:
# #     size_of_sq = int(500000 * (y_val/height) ** 16)
#     size_of_sq = int(30000 * (y_val/height) ** 8)
#     window_list = slide_window(img, x_start_stop=[0, width+size_of_sq], y_start_stop=[y_val,y_val+2*size_of_sq], 
#                     xy_window=(size_of_sq, size_of_sq), xy_overlap=(overlap,overlap))
#     rectangles.extend(window_list)

#     y_val = y_val + (1-overlap)*size_of_sq

print("num rectangles = ", len(rectangles))
labeled_img = draw_rectangles(img,rectangles)
plt.imshow(labeled_img)
plt.show()
plt.imshow(img)
plt.show()


# In[240]:

# Create the heat map
CV_FILLED = -1

detected_img = img.copy()
heat_img = np.zeros_like(img)
heat_map = np.zeros_like(img)
if cspace_val != 'RGB':
    if cspace_val == 'HSV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif cspace_val == 'LUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    elif cspace_val == 'HLS':
        feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    elif cspace_val == 'YUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
else: 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to rgb
    feature_image = np.copy(img)     
            
for rectangle in rectangles:
    pt1 = rectangle[0]
    pt2 = rectangle[1]
    crop_img = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
    size = (64,64)
    crop_img = cv2.resize(crop_img, size)

    img_features =[]
    spatial_features = bin_spatial(crop_img, size=spatial_size_val)
    hist_features = color_hist(crop_img, nbins=hist_bins_val, bins_range=hist_range_val)
    hog_features = get_hog_features(crop_img[:,:,hog_channel_val], orient=orient_val, 
                    pix_per_cell=pix_per_cell_val, cell_per_block=cell_per_block_val, vis=False, feature_vec=True)
#     img_features.append(hog_features)
    img_features.append(np.concatenate((spatial_features, hist_features, hog_features)))
#     img_features.append(np.concatenate((hist_features, hog_features)))
    X = np.vstack((img_features)).astype(np.float64)

#     X = X.reshape(1, -1)
    scaled_X = X_scaler.transform(X)
    prediction = svc.predict(scaled_X.reshape(1, -1))
    if prediction == 1:
        cv2.rectangle(detected_img, pt1, pt2, (255,255,255),thickness=4)
        cv2.rectangle(heat_img, pt1, pt2, color=(255,0,0), thickness=CV_FILLED)
        heat_map = cv2.addWeighted(heat_map, 0.95, heat_img, 0.05, 0)

final_img = cv2.addWeighted(img, 0.5, heat_map, 0.5, 0)

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.imshow(img)
ax1.set_title('original image')
ax2.imshow(detected_img)
ax2.set_title('Detected img')
ax3.imshow(heat_map)
ax3.set_title('Heat map')
ax4.imshow(final_img)
ax4.set_title('Final image')
plt.show()


# In[ ]:




# In[ ]:



