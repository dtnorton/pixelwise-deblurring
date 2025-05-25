# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 23:53:40 2025

@author: dtnor
"""
import cv2
import numpy as np

def apply_blur(image, kernel_size):
    """Applies a simple average blur to the image."""
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image

def wiener_filter(blurred_image, kernel, noise_power):
    """Applies the Wiener filter to deblur the image."""
    kernel = np.flipud(np.fliplr(kernel))
    dummy = np.copy(blurred_image)
    dummy = np.fft.fft2(dummy)
    kernel = np.fft.fft2(kernel, s=blurred_image.shape)
    kernel = np.conj(kernel) / (np.abs(kernel)**2 + noise_power)
    dummy = dummy * kernel
    deblurred_image = np.abs(np.fft.ifft2(dummy))
    return deblurred_image

# Load an example image
image_path = 'path/to/your/image.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply blur
kernel_size = 5  # Adjust the kernel size as needed
blurred_image = apply_blur(image, kernel_size)

# Deblur using Wiener filter
noise_power = 0.01  # Adjust the noise power as needed
deblurred_image = wiener_filter(blurred_image, np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size), noise_power)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Blurred Image', blurred_image)
cv2.imshow('Deblurred Image', deblurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
