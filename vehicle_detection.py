
# coding: utf-8

# In[193]:

# import methods

# from methods import plot3d
# from methods import bin_spatial
# from methods import color_hist
# from methods import get_hog_features
# # from methods import extract_features
# from methods import slide_window

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from skimage.feature import hog
# Import everything needed to edit/save/watch video clips

from IPython.display import HTML
from moviepy.editor import VideoFileClip

# NOTE: the next import is only valid 
# for scikit-learn version <= 0.17
# if you are using scikit-learn >= 0.18 then use this:
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split


# In[194]:

class CarDetector:
   
    def __init__(self, cspace='RGB', spatial_size=(32, 32),
                 hist_bins=32, hist_range=(0, 256), orient=9, 
                 pix_per_cell=8, cell_per_block=2, hog_channel=0):
        
        self.cspace =cspace
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.hist_range=hist_range
        self.orient = orient
        self.pix_per_cell=pix_per_cell
        self.cell_per_block=cell_per_block
        self.hog_channel=hog_channel
        self.vis=False
        self.feature_vec=True
        self.classifier = []
        self.scaler = []
        
    def get_classifier(self):
        return self.classifier
    
    def get_scaler(self):
        return self.scaler

    def train_classifier(self,cars,notcars):
        classifier_file = 'classifier.pkl'
        scaler_file = 'scaler.pkl'
        # extract features from dataset
        car_features = car_detector.extract_features(cars)
        print("Car features extracted")
        notcar_features = car_detector.extract_features(notcars)
        print("Other features extracted")# Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        print("X_scaler ready")
        #save the model
        joblib.dump(X_scaler, scaler_file)
        # Apply the scaler to X - normalise data
        scaled_X = X_scaler.transform(X)
        print("Data normalised")
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
        print(svc)
        # save the model
        joblib.dump(svc, classifier_file)
        self.classifier = svc
        self.scaler = X_scaler
        print("Model saved as:",classifier_file," and ", scaler_file)
    
    def load_classifier(self,classifier_file,scaler_file):
        self.classifier = joblib.load(classifier_file)
        self.scaler = joblib.load(scaler_file)
        print("Model loaded: \n\n",self.classifier)
    
    # Define a function to compute binned color features  
    def bin_spatial(self, img):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img,self.spatial_size).ravel() 
        # Return the feature vector
        return features
    
    # Define a function to compute color histogram features  
    def color_hist(self,img):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=self.hist_bins, range=self.hist_range)
        channel2_hist = np.histogram(img[:,:,1], bins=self.hist_bins, range=selfhist_range)
        channel3_hist = np.histogram(img[:,:,2], bins=self.hist_bins, range=self.hist_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features
    
    # Define a function to return HOG features and visualization
    def get_hog_features(self,img):
        # Call with two outputs if vis==True
        if vis == True:
            features, hog_image = hog(img, orientations=self.orient, pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                      cells_per_block=(self.cell_per_block, self.cell_per_block), transform_sqrt=True, 
                                      visualise=self.vis, feature_vector=self.feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:      
            features = hog(img, orientations=orient, pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                           cells_per_block=(self.cell_per_block, self.cell_per_block), transform_sqrt=True, 
                           visualise=self.vis, feature_vector=self.feature_vec)
            return features

    def extract_features_img(self, rgb_img):
    # apply color conversion if other than 'RGB'
        if self.cspace != 'RGB':
            if self.cspace == 'HSV':
                feature_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
            elif self.cspace == 'LUV':
                feature_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LUV)
            elif self.cspace == 'HLS':
                feature_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HLS)
            elif self.cspace == 'YUV':
                feature_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YUV)
        else: 
    #             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert to rgb
                feature_image = np.copy(rgb_img)
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=self.spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=self.hist_bins, bins_range=self.hist_range)
        # Call get_hog_features() with vis=False, feature_vec=True
        hog_features_0 = get_hog_features(feature_image[:,:,0], self.orient, 
                        self.pix_per_cell, self.cell_per_block, vis=False, feature_vec=True)
        hog_features_1 = get_hog_features(feature_image[:,:,1], self.orient, 
                        self.pix_per_cell, self.cell_per_block, vis=False, feature_vec=True)
        hog_features_2 = get_hog_features(feature_image[:,:,2], self.orient, 
                        self.pix_per_cell, self.cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features= np.concatenate((spatial_features, hist_features, hog_features_0,hog_features_1, hog_features_2))
        return features
            
    # Define a function to extract features from a list of images
    # Have this function call bin_spatial() and color_hist()
    # this function combines color, histogram and hog features extraction
    def extract_features(self,img_files):
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in img_files:
            # Read in each one by one
            rgb_img = mpimg.imread(file)
    #         image = cv2.imread(file) # reads a file into bgr values 0-255
            features.append(self.extract_features_img(rgb_img))
        
        # Return list of feature vectors
        return features
    
    # Define a function that takes an image,
    # start and stop positions in both x and y, 
    # window size (x and y dimensions),  
    # and overlap fraction (for both x and y)
    def slide_window(self,img, x_start_stop=[None, None], y_start_stop=[None, None], 
                        xy_window=(64, 64), xy_overlap=(0.5, 0.5),max_y=780):
        height, width, channels = img.shape
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched    
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_windows = np.int(xspan/nx_pix_per_step) - 1
        ny_windows = np.int(yspan/ny_pix_per_step) - 1
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = int(xs*nx_pix_per_step + x_start_stop[0])
                endx = int(startx + xy_window[0])
                starty = int(ys*ny_pix_per_step + y_start_stop[0])
                endy = int(starty + xy_window[1])
                # Append window position to list
                if endy<height and endx < width and endy<max_y:
                    window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list


# In[195]:

def load_dataset():
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
    return cars,notcars

def peak_data(cars, notcars):
    data_info = data_look(cars, notcars)
    print('Your function returned a count of', 
          data_info["n_cars"], ' cars and', 
          data_info["n_notcars"], ' non-cars')
    print('of size: ',data_info["image_shape"], ' and data type:', 
          data_info["data_type"])
    # Just for fun choose random car / not-car indices and plot example images   
    car_ind = np.random.randint(0, len(cars))
    notcar_ind = np.random.randint(0, len(notcars))

    # Read in car / not-car images
    car_image = mpimg.imread(cars[car_ind])
    notcar_image = mpimg.imread(notcars[notcar_ind])

    # Plot the examples
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(car_image)
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(notcar_image)
    plt.title('Example Not-car Image')
    plt.show()

# Define a function to return some characteristics of the dataset 
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    example_img = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    # Return data_dict
    return data_dict


# # Method
# * Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
# * Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
# * Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
# * Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
# * Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
# * Estimate a bounding box for vehicles detected.

# # Combined Color, Histogram and HOG Classification

# In[196]:

new_model = False
new_model = True

# create a CarDetector object
car_detector = CarDetector(cspace='YUV',hog_channel=1) # Options: RGB, HSV, LUV, HLS, YUV 
        
if (new_model):
    # load the dataset
    cars,notcars = load_dataset()
    peak_data(cars,notcars)
    #train the classifier
    car_detector.train_classifier(cars,notcars)
else:
    # load existing model
    car_detector = CarDetector()
    car_detector.load_classifier('classifier.pkl','scaler.pkl')


# In[ ]:

# RGB test accuracy = 0.9788 and low number of false positives in test images
# HSV test accuracy = 0.9777 but lots of false positives
# HLS test accuracy = 0.9688 but lots of false positives
# LUV test accuracy = 0.984  but lots of false positives
# YUV test accuracy = 0.9786 but lots of false positives  


# # Sliding Window Implementation

# In[169]:

def draw_rectangles(img,window_list,color= (255,255,255)):
    labeled_img = img.copy()
    for window in window_list:
        pt1 = window[0]
        pt2 = window[1]
        thickness = 4
        cv2.rectangle(labeled_img, pt1, pt2, color, thickness)
    return labeled_img


# In[170]:

# create a list of rectangles with different sizes across the lower part of the image for searching cars
def create_list_rectangles(img):
    
    height, width, channels = img.shape
    window_list=();
    rectangles = []

    step_h = 32
    start_h = step_h#int(height/4)
    stop_h = height 
    size_of_sq = int(256 * (1/height))
    y_val = int(9*height/16) 

#     size_vec = [64, 96, 128, 160]
#     overlap_vec = [0, 0.5, 0.65, 0.8]
    size_vec = [128, 160, 192, 224]
    overlap_vec = [0.5, 0.5, 0.5, 0.5]
    for i in range(len(size_vec)):
        size = size_vec[i]
        overlap = overlap_vec[i]
        window_list = slide_window(img, x_start_stop=[0, width+size], y_start_stop=[y_val,y_val+4*size], 
                        xy_window=(size, size), xy_overlap=(overlap,overlap),max_y=height*0.9)
        rectangles.extend(window_list)
    return rectangles


# In[171]:

# Create the heat map
def get_heat_map(img,rectangles,car_detector):
    
    heat_increment = 20
    CV_FILLED = -1
    heat_map = np.zeros_like(img)
    
    for rectangle in rectangles:
        heat_img = np.zeros_like(img)
        pt1 = rectangle[0]
        pt2 = rectangle[1]
        crop_img = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
        size = (64,64)
        crop_img = cv2.resize(crop_img, size)#.astype(np.float64)
        
        img_features = car_detector.extract_features_img(crop_img)

        features = np.vstack((img_features)).astype(np.float64)
        features = np.array(features).reshape(1, -1)
        
        feature_scaler = car_detector.get_scaler()
        classifier = car_detector.get_classifier()
        
        scaled_features = feature_scaler.transform(features)
        prediction = classifier.predict(scaled_features.reshape(1, -1))
        
        if prediction == 1:
            cv2.rectangle(heat_img, pt1, pt2, color=(heat_increment,0,0), thickness=CV_FILLED)
            heat_map = cv2.add(heat_map, heat_img)
            
    return heat_map


# In[172]:

# apply filter to the heat_map
# Note: th_ratio should be a ratio (0-1)
# It will be used with respect to the maximum pixel value in the image
def filter_heat_map(heat_map, th_ratio=0.5):
    red_channel = np.copy(heat_map[:,:,0])
#     print("max:",np.amax(red_channel))
    th = np.amax(red_channel)*th_ratio # define threshold
#     print("th:",th)
    filt_heat_map = np.zeros_like(heat_map)
    if np.amax(red_channel)>0:
        red_channel[red_channel>=th]=255
        red_channel[red_channel<th]=0
        filt_heat_map[:,:,0]=red_channel
    return filt_heat_map


# In[173]:

# computes positions and bounding rectangles identifying the location of detected vehicles
def get_detected(heat_map,area_th = 20):
     # define a threshold for minimum area required to be a positive detection
    imgray = heat_map[:,:,0]#cv2.cvtColor(heat_map,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray.astype(np.uint8),130,255,cv2.THRESH_BINARY)
    
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#     im2, contours, hierarchy = cv2.findContours(heat_map[:,:,0],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    detected_car_pos = [];
    detected_car_rectangles = [];
    for contour in contours:
        area = cv2.contourArea(contour)
#         print(area)
        if area>area_th:
            x,y,w,h = cv2.boundingRect(contour)
            M = cv2.moments(contour)
            # calculate image centroid
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            pt1 = (x,y)
            pt2 = (x+w,y+h)
            detected_car_pos.append([cx,cy])
            detected_car_rectangles.append([pt1,pt2])
    return detected_car_pos,detected_car_rectangles


# In[174]:

def process_image(img,debug=0):
    
    if not hasattr(process_image, "heat_map_old"):
        process_image.heat_map_old = np.zeros_like(img)
    
    if not process_image.heat_map_old.size:
        process_image.heat_map_old = np.zeros_like(img)
    
    decay = 0.2
    # apply decay to the heat_map
    process_image.heat_map_old = process_image.heat_map_old*(1-decay)
    
    rectangles = create_list_rectangles(img)

    heat_map = get_heat_map(img,rectangles,process_image.car_detector)
    
    filtered_heat_map = filter_heat_map(heat_map,th_ratio=0.1)
    
    process_image.heat_map_old = (filtered_heat_map*0.5 + process_image.heat_map_old*0.5)

    detected_car_pos,detected_car_rectangles = get_detected(process_image.heat_map_old,area_th = 1000)
    
    detected_cars_img = draw_rectangles(img,detected_car_rectangles,color= (255,255,255))
    if debug:
        # Ploting images
        labeled_img = draw_rectangles(img,rectangles)

        fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(24, 9))
        fig.tight_layout();
        

        ax1.imshow(heat_map)
        ax1.set_title('Heat map')
        ax2.imshow(filtered_heat_map.astype(np.uint8))
        ax2.set_title('Filtered heat_map')
        ax3.imshow(process_image.heat_map_old.astype(np.uint8))
        ax3.set_title('Used heat_map')
        ax4.imshow(img)
        ax4.set_title('Source image')
        ax5.imshow(labeled_img)
        ax5.set_title('Positive cars')
        ax6.imshow(detected_cars_img)
        ax6.set_title('Confirmed cars')
        ax1.axis('off');
        ax2.axis('off');
        ax3.axis('off');
        ax4.axis('off');
        ax5.axis('off');
        ax6.axis('off');
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.);
        plt.show();
        
    return detected_cars_img


# In[175]:

#reset heat_map_old
process_image.heat_map_old=np.zeros_like(img)
process_image.car_detector = car_detector


# In[179]:

# pipeline
# Implement a sliding-window technique and use your trained classifier to search for vehicles in images.

img = mpimg.imread('test_images/test6.jpg') # two cars, black and white
img = mpimg.imread('test_images/test5.jpg') # two cars, black and white
# img = mpimg.imread('test_images/test4.jpg') # two cars, black and white
# img = mpimg.imread('test_images/test3.jpg') # white car
# img = cv2.imread('test_images/test2.jpg') # no cars
# img = cv2.imread('test_images/test1.jpg') # two cars, black and white
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# img = img.astype(np.float32)
process_image(img,debug=1);


# In[153]:

video_output = 'small_video_output.mp4';
clip1 = VideoFileClip("small_video.mp4");
video_clip = clip1.fl_image(process_image); #NOTE: this function expects color images!!
get_ipython().magic('time video_clip.write_videofile(video_output, audio=False);')
print('Finished processing video file')


# In[ ]:



