## A program that can detect the hand, segment the hand, and count the number of fingers.

1. importing the necessary libraries


```python
import cv2
import numpy as np
from sklearn.metrics import pairwise
```

2. Declaring global variables for the background and ROI settings


```python
background = None
accumulated_weight = 0.5

roi_top = 20
roi_bottom = 300
roi_right = 300
roi_left = 600
```

* The program begins by defining a Region of Interest (ROI) within the video frame where the hand will be placed. This is crucial as it helps to focus on a specific part of the frame, making the processing faster and more efficient.
* This defines the coordinates of the ROI, which are later used to crop the hand region from the video frame.

3. Function to calculate the weighted average for the background


```python
def calc_accum_avg(frame, accumulated_weight):
    global background
    if background is None:
        background = frame.copy().astype("float")
        return None
    cv2.accumulateWeighted(frame, background, accumulated_weight)
```

* Background Calculation:
    The first 60 frames of the video feed are used to calculate the average background. This background is later subtracted from subsequent frames to isolate the hand from the background.

* Weighted Average:
The function cv2.accumulateWeighted() is used here. This function accumulates a weighted average of the input frame, which helps to smooth out the noise and get a cleaner background image.

* background is initialized as None, and in the first iteration, it's set to the first frame. In subsequent frames, cv2.accumulateWeighted() updates the background by adding a weighted sum of the new frame, which smooths out the background over time.

4. Function to segment the hand region from the background


```python
def segment(frame, threshold=25):
    global background
    diff = cv2.absdiff(background.astype("uint8"), frame)
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    else:
        hand_segment = max(contours, key=cv2.contourArea)
        return (thresholded, hand_segment)
```

The segment function isolates the hand from the background in a video frame:

Parameters:

* frame: The current video frame.

* threshold: The threshold value for binary segmentation (default is 25).

Background Subtraction:

* cv2.absdiff calculates the absolute difference between the background and the current frame to highlight changes.

Binary Thresholding:

* cv2.threshold converts the difference image to a binary image where changes are white (255) and unchanged areas are black (0).

Contour Detection:

* cv2.findContours identifies contours in the binary image.
Only external contours are retrieved (cv2.RETR_EXTERNAL), and contours are simplified (cv2.CHAIN_APPROX_SIMPLE).
Return Values:

If contours are found, the largest contour (assumed to be the hand) is selected.

* Returns a tuple containing:
thresholded: The binary image.
hand_segment: The largest contour (the hand).

5. Function to count the fingers based on the segmented hand


```python
def count_fingers(thresholded, hand_segment):
    conv_hull = cv2.convexHull(hand_segment)
    top = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
    left = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
    right = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])

    cX = (left[0] + right[0]) // 2
    cY = (top[1] + bottom[1]) // 2

    distance = pairwise.euclidean_distances([(cX, cY)], Y=[left, right, top, bottom])[0]
    max_distance = distance.max()

    radius = int(0.8 * max_distance)
    circumference = (2 * np.pi * radius)

    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
    cv2.circle(circular_roi, (cX, cY), radius, 255, 10)

    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
    contours, _ = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    count = 0
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        out_of_wrist = ((cY + (cY * 0.25)) > (y + h))
        limit_points = ((circumference * 0.25) > cnt.shape[0])
        if out_of_wrist and limit_points:
            count += 1

    return count


```

Count Fingers Function Overview
The count_fingers function estimates the number of fingers in the hand region:

Convex Hull Computation:

* cv2.convexHull calculates the convex hull of the hand contour, which is the smallest convex shape enclosing the hand.

Determine Extreme Points:

* Identifies the top, bottom, left, and right extremes of the convex hull by finding minimum and maximum coordinates.

Calculate Center and Radius:

* cX and cY are the center of the hand, computed by averaging the extreme points.

* pairwise.euclidean_distances measures distances from the center to the extreme points.

* max_distance is the largest of these distances.

* radius is set to 80% of max_distance, and circumference is calculated based on this radius.

Create Circular ROI:

* A blank image is created, and a circle with the computed radius is drawn to define a circular region of interest (ROI) around the hand.

Apply ROI Mask:

* cv2.bitwise_and applies the circular ROI mask to the thresholded image.
cv2.findContours retrieves contours from the masked image.

Count Fingers:

* For each contour, cv2.boundingRect obtains the bounding rectangle.
out_of_wrist checks if the contour is above the wrist.
limit_points ensures the contour has enough points.
Contours meeting both criteria are counted as fingers.

6. Running the program


```python

cam = cv2.VideoCapture(0)
num_frames = 0

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()

    roi = frame[roi_top:roi_bottom, roi_right:roi_left]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    if num_frames < 60:
        calc_accum_avg(gray, accumulated_weight)
        if num_frames <= 59:
            cv2.putText(frame_copy, "WAIT! GETTING BACKGROUND AVG.", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Finger Count", frame_copy)
    else:
        hand = segment(gray)
        if hand is not None:
            thresholded, hand_segment = hand
            cv2.drawContours(frame_copy, [hand_segment + (roi_right, roi_top)], -1, (255, 0, 0), 1)
            fingers = count_fingers(thresholded, hand_segment)
            cv2.putText(frame_copy, f"Fingers: {fingers}", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Thresholded", thresholded)

    cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0, 0, 255), 5)
    num_frames += 1

    cv2.imshow("Finger Count", frame_copy)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()

```

* Opens the default camera (webcam).

* Continuously captures frames from the webcam.

* Flips the frame horizontally and creates a copy for display purposes.

* Extracts a region of interest from the frame where hand detection will occur.

* Converts the ROI to grayscale and applies Gaussian blur.

* Accumulates the background model over the first 60 frames.

* Segments the hand and counts fingers once background accumulation is complete.

* Draws contours, displays the number of fingers, and shows the frame.
