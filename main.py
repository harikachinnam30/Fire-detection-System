import cv2
import numpy as np
from imutils import contours
from skimage import measure
import imutils

def detect_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Thresholding and noise reduction
    thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    # Label connected components in the thresholded image
    labels = measure.label(thresh, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")

    # Loop over the unique labels
    for label in np.unique(labels):
        if label == 0:
            continue

        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        if numPixels > 200:
            mask = cv2.add(mask, labelMask)

    # Find contours in the mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts)[0]

    # Loop over the contours and draw them on the image with different colors
    color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255)]  # Colors for each contour
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        color = color_list[i % len(color_list)]  # Cycle through colors if more than 5 contours
        cv2.circle(image, (int(cX), int(cY)), int(radius), color, 3)
        cv2.putText(image, "#{}".format(i + 1), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    return image, thresh, mask

# Load the image
image_path = "c:/Users/priya/Downloads/Fire-Detection-Image-Processing-master/Source Code/forestfire.jpg"
img = cv2.imread(image_path)

# Step 1: Thresholding
thresholded_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresholded_img = cv2.GaussianBlur(thresholded_img, (11, 11), 0)

# Apply different thresholding techniques
thres1 = cv2.threshold(thresholded_img, 150, 255, cv2.THRESH_BINARY)[1]   # bg separation
thres2 = cv2.threshold(thresholded_img, 150, 255, cv2.THRESH_BINARY_INV)[1]  # highlights bg instead
thres3 = cv2.threshold(thresholded_img, 150, 255, cv2.THRESH_TRUNC)[1] # supress high intensity egions
thres4 = cv2.threshold(thresholded_img, 150, 255, cv2.THRESH_TOZERO)[1] # to remove bg
thres5 = cv2.threshold(thresholded_img, 150, 255, cv2.THRESH_TOZERO_INV)[1]  # supress brigher regions{while keeping bg}

# Step 2: Watershed Segmentation
thresh = cv2.threshold(thresholded_img, 150, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=4)
labels = measure.label(thresh, background=0)
mask = np.zeros(thresh.shape, dtype="uint8")

# Step 3: Contour Detection
final_img, thresholded_img, mask = detect_contours(img.copy())

# Show images sequentially
cv2.imshow("BINARY Threshold", thres1)
# cv2.imshow("BINARY_INV Threshold", thres2) 
# cv2.imshow("TRUNC Threshold", thres3)
# cv2.imshow("TOZERO Threshold", thres4)
# cv2.imshow("TOZERO_INV Threshold", thres5)
cv2.imshow("Watershed Segmentation", mask)
cv2.imshow("Final Image with Contours", final_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
