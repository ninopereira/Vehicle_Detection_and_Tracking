# Vehicle Detection and Tracking

![Sample](output_images/report_img/result.png)

This is a short report containing a brief description of the main strategies and methods employed in solving the problem of detecting vehicles in a video stream image.

#### See Video Result [here](https://youtu.be/Fkxe-Hxgbqw)


## Goals
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[im1]: ./output_images/report_img/car_not_car.png
[im2]: ./output_images/report_img/hog.png
[im3]: ./output_images/report_img/orientations_acc.png
[im4]: ./output_images/report_img/pix__cell.png
[imHOG]: ./output_images/report_img/HOG_param.png
[im5]: ./output_images/report_img/spatial_feat.png
[im6]: ./output_images/report_img/hist_feat.png
[im7]: ./output_images/report_img/profile_extraction.png

[im8]: ./output_images/report_img/test_acc_extr_time.png
[im9]: ./output_images/report_img/sliding_window.png
[im10]: ./output_images/report_img/pipeline2.png
[im11]: ./output_images/report_img/result.png

[video1]: ./output_images/project_output.mp4


## Histogram of Oriented Gradients (HOG)

### 1. Extraction of HOG features from the training images.


The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `vehicle_detection.ipynb`).  

```javascript
def get_hog_features(self,img_channel):
	features = hog(img_channel, orientations=self.orient, pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),cells_per_block=(self.cell_per_block, self.cell_per_block), transform_sqrt=True, visualise=self.vis, feature_vector=self.feature_vec)
	return features
```
I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][im1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HSV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][im2]

### 2. Final choice of HOG parameters.

I tried various combinations of parameters, including changing the number of orientations (6, 9 and 12) and testing the effect on the performance of the svm prediction in the test set. In this case using 9 orientations seems to work slightly better.

![alt text][im3]

Next I varied the number of pixels per cell (8 or 16) and combined with a different number of cells per block (2 or 4).

![alt text][imHOG]

As shown above the best combination for the tested scenarios is using 8 pixels per cell and 2 cells per block, yielding a better accuracy in the classification of the test set.

#### Feature extraction methods
I investigated using other features as spatial (which uses the entire color profile of the image) and the histogram (which groups colors in bins).

![alt text][im5]
![alt text][im6]

I profiled the time it takes to extract features using these 3 different methods and concluded that extracting hog features is by far the most expensive method. It takes double the time to extract hog features from one single channel than extracting histogram features from all the 3 channels in an image.
The time to extract spatial features is negligible.
![alt text][im7]

We want to combine the methods in such a way that the classifier gets sufficient information to compute an accurate prediction as well being able to do it fast. That's why it is important to consider the extraction time as a fundamental part of the process.
I tested 30 different combinations of extraction methods accross a variety of color spaces and varying the channels used for hog extraction.
Here are the results for the top 10:
![alt text][im8]

By looking at the results we can see that the *HSV_spa_hist_hog2* (combining HSV spatial, histogram and hog on V channel) together with *HSV_his_hog_all* (combining HSV histogram and hog on the 3 channels) are the methods with higher accuracy (0.99) on the test set. However the former outperforms the latter by far when looking at the computational time required to extract the features. As such it was decided to proceed with this *HSV_spa_hist_hog2* combination of features.


### 3. How to train the classifier using the selected features

To handle the feature extraction and training of a classifier I created a class named CarDetector which has all the parameters defined and loaded in the __init method. It comprises all the methods necessary for feature extraction including color spatial features, histograms and hog features. I used a linear SVM using the default parameters.
All the features (spatial, histogram and hog) previously described were extracted from the training data and used to train the svm classifier. Features were scaled to zero mean and unit variance before training the classifier using the StandardScaler() method.

``` javascript
car_features = car_detector.extract_features(cars)
notcar_features = car_detector.extract_features(notcars)
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

svc = LinearSVC()
svc.fit(X_train, y_train)
```

## Sliding Window Search

### 1. Parameters for the sliding window search: scales and overlap

The number and distribution of windows across the image are very important parameters as they influence the quality and the speed of detection of vehicles in the image.
First of all it was decided not to search for vehicles in the upper part of the image as it includes mostly the horizon and sky which are not relevant for the task.
Then, through experimentation, by varying the number, size and overlap of windows it was decided to use up to 3 rows (depending if they fit in the image or not) of two different sizes of squares filling the lower region of the image and overlapping each other by 50%, giving a total of 106 rectangles. Considering that feature extraction takes 1.82ms and predicting takes about 1ms we spend at least 300ms just on searching vehicles in one single image.

![alt text][im9]

Note that special care was taken in not considering the lower part of the image because it contains part of our car. Also, the number of rectangles spans the entire width of the camera from left to right.

### 2. Pipeline description.  Minimize false positives and reliably detect cars.

As described before the feature extraction and training of the classification method was taken very careful in order to assure a good detection and low number of false positives.
The pipeline works as follows. The source image (1) is taken from the video stream. A search is conducted in the image using all the rectangles previously defined by the sliding window (2). Then, the ones which the classifier considers to be positive results (3) are converted to a first temporary heat map (4). From this provisional heat map only the strongest signals are then filtered (5) and prepared for merge with the existing heat map in a ratio of 20% to 80%. After merging, the final heat (6) map is then further filtered by considering only the most heated regions (above 100 rgb values) to be positive.
Note that the heat map conserves it state from iteration to iteration and has a decay factor of about 5% in each frame.

![alt text][im10]

The result shows only regions of high certainty where vehicles are being detected.
![alt text][im11]

---

##  Video Implementation

### 1. Final video output.  

The pipeline works quiet well on identifying positive cars. There are almost no false positives although sometimes due to the nature of the implemented filter, there are some failures in detecting the vehicles continuously.
Here's a [link to my video result](./output_images/project_output.mp4)


### 2. Filter for false positives and combining overlapping bounding boxes in a heat map.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heat map and then threshold that map to identify vehicle positions.

In the pipeline the filtering part comprises:

 1. Applying the decay to existing heat map (*heat_map_old*)
 2. Detect positive and create a new heat_map
 3. Filter the new heat map so that only the high confidence regions (*valid_data*) remain
 4. Get the centroids, rectangles and areas of those regions
 5. Given these regions *valid_data* eliminate false positives from the new heat map
 6. Filter the new heat_map again but now keeping all the detected areas
 7. Add the detected areas to the old heat map (*heat_map_old*) in a ratio of 20% to 80%
 8. From the updated old heat map (*heat_map_old*) filter the high confidence regions and use them to identify positive vehicles in the image.

```javascript
decay = 0.05
#apply decay to the heat_map
process_image.heat_map_old = process_image.heat_map_old*(1-decay)

#get heat_map
heat_map = get_heat_map(img,process_image.rectangles,process_image.car_detector,heat_increment)

#filter the heat map to get rid of false positives
filtered_heat_map = filter_heat_map(heat_map,th_ratio=0.5)
valid_data = get_detected(filtered_heat_map,area_th = 1000,heat_th=heat_thres)

#now that we know the location of valid positives
#we can use the original heat map to get the complete area and filter out false positives
filtered_heat_map = filter_heat_map(heat_map,th_ratio=0.05)
data = get_detected(filtered_heat_map,area_th = 1000,heat_th=heat_thres)
posit_data = filter_by_location(data, valid_data)

#now we create a new filtered_heat_map with the posit_data only
filtered_heat_map = create_map_from_data(img,posit_data,heat_increment=255)

process_image.heat_map_old = (filtered_heat_map*0.2 + process_image.heat_map_old*0.8)

final_data = get_detected(process_image.heat_map_old, area_th=1000, heat_th=100)

detected_car_rectangles = []
for elem in final_data:
detected_car_rectangles.append(elem[1])
    
```

The function (*get_detected*) that performs the merging of overlapping squares in the image uses the opencv libraries for finding contours (*cv2.findContours*), computing areas (*cv2.contourArea(contour)*) create bounding boxes (*cv2.boundingRect(contour)*) and calculate the centroids (*cv2.moments(contour)*).

```javascrip
def get_detected(heat_map,area_th = 20,heat_th=80):
     # define a threshold for minimum area required to be a positive detection
    imgray = heat_map[:,:,0]#cv2.cvtColor(heat_map,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray.astype(np.uint8),heat_th,255,cv2.THRESH_BINARY)
    
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    data = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area>area_th:
            x,y,w,h = cv2.boundingRect(contour)
            M = cv2.moments(contour)  # calculate image centroid
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            pt1 = (x,y)
            pt2 = (x+w,y+h)
            coord = [cx,cy]
            rect = [pt1,pt2]
            data.append((coord,rect,area))
    return data
```

## Discussion

### 1. Issues of this project and further work

This project is working relatively well on most of the video frames. There are very few number of false positives and for really close cars it performs the detection very accurately. There is still plenty room to improve on several aspects, though.
I soon realized that the bottleneck was in the feature extraction methods. Depending on the methods you chose you may compromise no only the quality but also the computation time required for classification.

I tried to accomplish good quality results by investigating the performance of several extraction methods and configuration parameters and I also addressed the computational time by profiling up to 30 different combinations of feature extraction methods.

In the end I chose a HSV color space in conjunction with spatial, histogram and hog (on V channel) feature extractions as the preferred choice, yielding 99% accuracy on the test set with a quite low time for feature extraction in comparison to similar performing methods.

I didn't try to use other classifiers as this one was already performing quite well and quite fast for this dataset. But that's certainly something worth trying, especially if using a larger dataset.

On the filters, including managing and keeping the heat map, a lot was done, but it could be better tuned and maybe smoothed by making use of the area and centroid information we already have. This is definitely a piece of work to be refined in future work.  
