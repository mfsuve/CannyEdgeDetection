import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
import re

# Defining constants to be used in Double Thresholding
STRONG = 255
WEAK = -127

def get_gaussian_kernel(size, std):
	kernel = cv2.getGaussianKernel(size, std)
	return np.dot(kernel, kernel.T)


def conv(img, kernel):
	(w, h) = kernel.shape
	# new image with extended size for zero padding
	padded_img = np.zeros((W + w - 1, H + h - 1))
	# Assuming the kernel shape consists of only odd numbers
	# putting the image into the middle of the new image
	padded_img[w // 2:W + w // 2, h // 2:H + h // 2] = np.copy(img)
	# create new resulting image
	result = np.zeros((W, H))

	# Convolution operation
	for i in range(W):
		for j in range(H):
			result[i, j] = np.sum(np.multiply(padded_img[i:i + w, j:j + h], kernel[::-1, ::-1]))

	return result


def gradient(img):
	# Using the sobel kernels
	X_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
	Y_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
	return conv(img, X_kernel), conv(img, Y_kernel)


def gradient_and_suppression(img):
	X, Y = gradient(img)

	# Non-maximum suppression
	# Angles and Magnitudes of the gradients
	Angles = np.arctan2(Y, X) * 180 / np.pi
	Magnitudes = np.sqrt(np.square(X) + np.square(Y))

	result = np.zeros((W, H))

	# Loop for getting only the biggest values of the edges along the gradient (Suppression)
	for i in range(1, W - 1):
		for j in range(1, H - 1):
			angle = Angles[i, j]
			if np.abs(angle + 180) <= 22.5 or np.abs(angle) <= 22.5:  # Left or Right
				if Magnitudes[i, j - 1] < Magnitudes[i, j] and Magnitudes[i, j + 1] < Magnitudes[i, j]:
					result[i, j] = Magnitudes[i, j]
			elif np.abs(angle + 90) <= 22.5 or np.abs(angle - 90) <= 22.5:  # Down or Up
				if Magnitudes[i - 1, j] < Magnitudes[i, j] and Magnitudes[i - 1, j] < Magnitudes[i, j]:
					result[i, j] = Magnitudes[i, j]
			elif np.abs(angle - 45) <= 22.5 or np.abs(angle + 135) <= 22.5:  # UpRight or DownLeft
				if Magnitudes[i - 1, j + 1] < Magnitudes[i, j] and Magnitudes[i + 1, j - 1] < Magnitudes[i, j]:
					result[i, j] = Magnitudes[i, j]
			else:  # DownRight or UpLeft
				if Magnitudes[i + 1, j + 1] < Magnitudes[i, j] and Magnitudes[i - 1, j - 1] < Magnitudes[i, j]:
					result[i, j] = Magnitudes[i, j]

	return result


def double_threshold(img, up, down):
	min = np.min(img)
	max = np.max(img)
	# Normalized the image between [0, 1] to have better threshold values
	normalized = (img - min) / (max - min)

	# Thresholding
	normalized[normalized < down] = 0
	normalized[normalized >= up] = STRONG
	normalized[np.logical_and(normalized >= down, normalized < up)] = WEAK

	return normalized


def edge_tracking(img):
	def connect(i, j):
		for h in [-1, 0, 1]:
			for v in [-1, 0, 1]:
				if img[i - h, j - v] == WEAK:
					img[i - h, j - v] = STRONG
					# If the weak edge has been turned into strong edge,
					# Repeat this process for that new strong edge
					connect(i - h, j - v)

	for i in range(W):
		for j in range(H):
			if img[i, j] == STRONG:
				# Making strong all the weak edges connecting this strong edge recursively
				connect(i, j)

	# Clear all the remaining weak edges
	img[img == WEAK] = 0

	return img


parser = argparse.ArgumentParser()
parser.add_argument('image_path', type=str, help='Image path to be edge-detected')
parser.add_argument('--kernel_size', type=int, help='Kernel size for the Gaussian filter', default=3)
parser.add_argument('--std', type=float, help='Standard deviation of the Gaussian filter', default=-1)
parser.add_argument('--upper_threshold', type=float, help='Upper threshold value for double thresholding', default=.2)
parser.add_argument('--lower_threshold', type=float, help='Lower threshold value for double thresholding', default=.075)
args = parser.parse_args()

# Read the image
image = cv2.imread(args.image_path)
if image is None:
	print('Please give a valid image path')
	exit()
# Extracting the filename from the console argument
filename = re.findall(r"[\w']+", args.image_path)[-2]
# Extracting the size to use in functions above
W, H, _ = image.shape
# Turn it into grayscale image
image = np.average(image, axis=2, weights=[.0722, .7152, .2126])
# Smooth the image
# Gaussian Kernel
gaussian_kernel = get_gaussian_kernel(args.kernel_size, args.std)
image = conv(image, gaussian_kernel)
# Finding the gradient and doing the non-max suppression
image = gradient_and_suppression(image)
# Double Thresholding
image = double_threshold(image, args.upper_threshold, args.lower_threshold)
# Edge tracking by hysteresis
image = edge_tracking(image)

plt.title(filename + ' edges')
plt.imshow(image, cmap='gray')
plt.show()
cv2.imwrite(filename + '_edges.png', image)
