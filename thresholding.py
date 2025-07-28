# thresholding.py
import cv2
import numpy as np

def apply_thresholds(image):
    """
    This function applies multiple thresholding techniques on the input image.
    It returns a dictionary of thresholded images.
    """
    # Apply different thresholding techniques
    ret, thres1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    ret, thres2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thres3 = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
    ret, thres4 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
    ret, thres5 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)

    # Return the thresholded images as a dictionary or list
    thresholded_images = {
        'BINARY': thres1,
        'BINARY_INV': thres2,
        'TRUNC': thres3,
        'TOZERO': thres4,
        'TOZERO_INV': thres5
    }

    return thresholded_images
