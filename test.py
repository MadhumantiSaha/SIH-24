import cv2 as cv
import numpy as np

img = cv.imread('moon.jpg')
cv.imshow('Normal1', img)
img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
cv.imshow('Normal2', img)
cv.imshow('Normal', img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blurred_img = cv.GaussianBlur(gray, (5, 5), 0)
cv.imshow('Normal', img)
cv.imshow('Filtered', blurred_img)

canny_img = cv.Canny(img, 0, 255)
blur_canny = cv.Canny(blurred_img, 0, 255)
cv.imshow('Canny', canny_img)
cv.imshow('Blur Canny', blur_canny)

sharp = blurred_img-gray
cv.imshow('Sharp', sharp)
cv.imshow('Image', img)

mask = canny_img.astype(np.uint8)

new_img = cv.bitwise_and(img, img, mask=mask)

cv.imshow('New Image', new_img)

superimposed = cv.bitwise_xor(img,new_img)
cv.imshow('Superimposed', superimposed)



#Find minimum and maximum pixel values
min_val = np.min(gray)
max_val = np.max(gray)

scale = 255 / (max_val - min_val)
scale_factor = 1.0  # Adjust this factor to control the amount of scaling for low values
stretched = (gray - min_val) * scale * (1 + scale_factor * (gray / 255.0)) ** 0.5
stretched = np.uint8(stretched)

# Display the original and stretched images
cv.imshow('Original', gray)
cv.imshow('Stretched', stretched)
cv.imshow('Stretched_inverted', cv.bitwise_not(stretched))
cv.waitKey(0)



