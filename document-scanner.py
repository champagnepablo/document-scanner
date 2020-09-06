import sys
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt


def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect

def contourThresholding(img):
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(6, 4))
    img = cv2.medianBlur(img, 53)
    LabImage = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
    _, a, b = cv2.split(LabImage)
    noLightImage = (np.uint8) ((np.divide(a,2) +  np.divide(b,2))-1)
    histoImg = clahe.apply(noLightImage)
    thresoldedImage = cv2.adaptiveThreshold(histoImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 73, 2)
    return thresoldedImage

def sheetTransforming(img,originalImage, width):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,0.02*peri,True)
        croppedPoints = np.float32([[0,0],[originalImage.shape[0]-1, 0], 
                       [originalImage.shape[0]-1, originalImage.shape[1]-1], [0, originalImage.shape[1]-1]])
        approx2 = order_points(approx[:,0])
        transformationMatrix  = cv2.getPerspectiveTransform(approx2, croppedPoints)
        croppedImage = cv2.warpPerspective(originalImage,transformationMatrix,
                        (originalImage.shape[0]-1,originalImage.shape[1]-1))
        cv2.drawContours(originalImage, [approx], -1, (255), 55, cv2.LINE_AA)
        height = width * 4 / 3
        dsize = (width,  (int) (height))
        output = cv2.resize(croppedImage, dsize)
        return output
    else :
        print("Unable to get sheet contour. Cleaning image without transforming perspective")
        return originalImage

def cleanPainting(image):
    Lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    _, a, b = cv2.split(Lab)
    minB = np.mean(b) - (np.max(b)*0.03)
    _,thresh2 = cv2.threshold(b, (int) (minB),255,cv2.THRESH_BINARY)
    thresh2 = cv2.bitwise_not(thresh2)
    minA = np.mean(a) + (np.max(a)*0.02)
    _,thresh3 = cv2.threshold(a,(int) (minA),255,cv2.THRESH_BINARY)
    paintings = cv2.bitwise_or(thresh3,thresh2)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(image_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,11,2)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cleaned_inv = opening - paintings
    cleaned = cv2.bitwise_not(cleaned_inv)
    return cleaned,thresh

def main():
    image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
    originalImage = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
    img1 = contourThresholding(image)
    img = sheetTransforming(img1,image, 2440)
    clean, paintings = cleanPainting(img)
    if sys.argv[1] == "-S" :
        _, axarr = plt.subplots(2, 2)
        axarr[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axarr[0, 0].set_title('Original image with contours')
        axarr[0, 1].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        axarr[0, 1].set_title('Thresholded image')
        axarr[1, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axarr[1, 0].set_title('Thresholded image')
        axarr[1, 1].imshow(cv2.cvtColor(paintings, cv2.COLOR_BGR2RGB))
        axarr[1, 1].set_title('Paintings detected')
        plt.show()
    elif sys.argv[1] == "-F":
        _, axarr = plt.subplots(2)
        axarr[0].imshow(cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB))
        axarr[0].set_title('Original image')
        axarr[1].imshow(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
        axarr[1].set_title('Final result')
        plt.show()
    elif sys.argv[1] == "-R":
        plt.imshow(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
        plt.show()

if __name__ == "__main__":
    main()
