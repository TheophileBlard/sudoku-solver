import os

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def resize_img(img, width = None, height = None, inter = cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = img.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return img

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized_img = cv.resize(img, dim, interpolation = inter)

    # return the resized image
    return resized_img

def display_img(img, title='img'):
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv.Laplacian(image, cv.CV_64F).var()

def preprocess_img(resized):
    # 2. Grayscale
    gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
    #display_img(gray, title='gray')    

    # 3. Blur
    blurred = cv.bilateralFilter(
        gray, 
        d=7, 
        sigmaColor=50, 
        sigmaSpace=50
    )
    #display_img(blurred, title='blurred')

    # 4. Threshold
    thresh = cv.adaptiveThreshold(
        blurred, 
        maxValue=255, 
        adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
        thresholdType=cv.THRESH_BINARY_INV, 
        blockSize=11,
        C=2
    )

    # Dilate the image to increase the size of the grid lines.
    kernel = cv.getStructuringElement(cv.MORPH_CROSS,(3,3)) #np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]])
    preprocessed = cv.dilate(thresh, kernel)

    #display_img(thresh, title='thresh')

    # 5. Denoising (morphological operators)
    #preprocessed = cv.erode(thresh, None, iterations=1)
    #preprocessed = cv.dilate(preprocessed, None, iterations=1)
    #display_img(preprocessed, title='preprocessed')
    #preprocessed = thresh
    #display_img(preprocessed, title='preprocessed')
    return preprocessed

def find_grid_corners(preprocessed):
    contours, hierarchy = cv.findContours(preprocessed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #contours_img = img.copy()
    #cv.drawContours(contours_img, contours, -1, (0, 0, 255), 2)
    #cv.imshow('contours', contours_img)
    #cv.waitKey(0)

    # 2. Extract biggest contour
    biggest_cnt = sorted(contours, key=lambda c: cv.contourArea(c), reverse=True)[0]

    #approx_img = img.copy()
    #cv.drawContours(approx_img, [biggest], -1, (0, 255, 0), 2)
    #cv.imshow('approx', approx_img)
    #cv.waitKey(0)

    # 3. Determine its corners
    top_left = tuple(sorted(biggest_cnt, key=lambda p: p[0][0] + p[0][1])[0][0])
    top_right = tuple(sorted(biggest_cnt, key=lambda p: p[0][0] - p[0][1], reverse=True)[0][0])
    bottom_left = tuple(sorted(biggest_cnt, key=lambda p: p[0][0] - p[0][1])[0][0])
    bottom_right = tuple(sorted(biggest_cnt, key=lambda p: p[0][0] + p[0][1], reverse=True)[0][0])

    #corner_img = img.copy()
    #cv.circle(corner_img, top_left, 8, (0, 0, 255), -1)
    #cv.circle(corner_img, top_right, 8, (0, 255, 0), -1)
    #cv.circle(corner_img, bottom_left, 8, (255, 0, 0), -1)
    #cv.circle(corner_img, bottom_right, 8, (255, 255, 0), -1)

    corners = np.float32([
        top_left,
        top_right,
        bottom_left,
        bottom_right
    ])

    return corners

def warp_and_crop(img, img_corners):
    L = int(np.linalg.norm(img_corners[0]-img_corners[1])) # distance
    
    #int(np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))) # could do a mean
    dest_corners = np.float32([
        [0, 0],
        [L - 1, 0],
        [0, L - 1],
        [L - 1, L - 1],
    ])

    M = cv.getPerspectiveTransform(img_corners, dest_corners)
    return cv.warpPerspective(img, M, (L, L))

def infer_grid(img):
	squares = []
	side = img.shape[:1]
	side = side[0] / 9
	for j in range(9):
		for i in range(9):
			p1 = (int(i * side), int(j * side))  # Top left corner of a bounding box
			p2 = (int((i + 1) * side), int((j + 1) * side))  # Bottom right corner of bounding box
			squares.append((p1, p2))
	return squares

def preprocess_grid(resized):
    # 2. Grayscale
    gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
    #display_img(gray, title='gray')

    # 3. Blur
    #blurred = gray

    #blurred = cv.GaussianBlur(gray, (9, 9), 0)

    fm = variance_of_laplacian(gray)
    print(fm)

    if fm > 100:
        d = 11
        sigma = 90
    if fm > 50:
        d = 9
        sigma = 70
    elif fm > 10:    
        d = 7
        sigma = 50        
    else:
        d = 5
        sigma = 30

    blurred = cv.bilateralFilter(
            gray, 
            d=d, 
            sigmaColor=sigma, 
            sigmaSpace=sigma
        )
    
    
    #display_img(blurred, title='blurred')

    # 4. Threshold
    thresh = cv.adaptiveThreshold(
        blurred, 
        maxValue=255, 
        adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C, 
        thresholdType=cv.THRESH_BINARY_INV, 
        blockSize=11,
        C=2
    )
    display_img(thresh, title='thresh')

    """
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (40, 1)) # 23 => 1/9 de l'image ?
    horizontal_mask = cv.erode(thresh, horizontal_kernel, iterations=1)
    horizontal_mask = cv.dilate(horizontal_mask, horizontal_kernel, iterations=5)

    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 40)) # 23 => 1/9 de l'image ?
    vertical_mask = cv.erode(thresh, vertical_kernel, iterations=1)
    vertical_mask = cv.dilate(vertical_mask, vertical_kernel, iterations=5)

    preprocessed = cv.subtract(thresh, horizontal_mask)
    preprocessed = cv.subtract(preprocessed, vertical_mask)
    """
    # 5. Denoising (morphological operators)
    preprocessed = cv.erode(thresh, None, iterations=1)
    preprocessed = cv.dilate(preprocessed, None, iterations=1)
    display_img(preprocessed, title='preprocessed')
    preprocessed = thresh
    
    return preprocessed

#folder_path = '/home/theophile/Documents/Projects/repos/sudoku-image-solver/sudoku_images'
#img_name = 'sudoku3.jpg' source v
#img_path = os.path.join(folder_path, img_name)


#original = cv.imread('img/sudoku-original.jpg')
original = cv.imread('../repos/sudoku_dataset/images/image1020.jpg')
resized = resize_img(original, height = 500)

# 1. Find grid
preprocessed = preprocess_img(resized)
corners = find_grid_corners(preprocessed)
cropped = warp_and_crop(resized, corners)
display_img(cropped, title="cropped")

# 2. Extract digits
resized = resize_img(cropped, height = 500)
squares = infer_grid(resized)

grid = resized.copy()
for square in squares:
    grid = cv.rectangle(grid, tuple(x for x in square[0]), tuple(x for x in square[1]), (255, 0, 0))
display_img(grid, title="grid")

# test --------
def cut_from_rect(img, rect):
	"""Cuts a rectangle from an image using the top left and bottom right points."""
	return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]


def scale_and_centre(img, size, margin=0, background=0):
	"""Scales and centres an image onto a new background square."""
	h, w = img.shape[:2]

	def centre_pad(length):
		"""Handles centering for a given length that may be odd or even."""
		if length % 2 == 0:
			side1 = int((size - length) / 2)
			side2 = side1
		else:
			side1 = int((size - length) / 2)
			side2 = side1 + 1
		return side1, side2

	def scale(r, x):
		return int(r * x)

	if h > w:
		t_pad = int(margin / 2)
		b_pad = t_pad
		ratio = (size - margin) / h
		w, h = scale(ratio, w), scale(ratio, h)
		l_pad, r_pad = centre_pad(w)
	else:
		l_pad = int(margin / 2)
		r_pad = l_pad
		ratio = (size - margin) / w
		w, h = scale(ratio, w), scale(ratio, h)
		t_pad, b_pad = centre_pad(h)

	img = cv.resize(img, (w, h))
	img = cv.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv.BORDER_CONSTANT, None, background)
	return cv.resize(img, (size, size))


def find_largest_feature(inp_img, scan_tl=None, scan_br=None):
	"""
	Uses the fact the `floodFill` function returns a bounding box of the area it filled to find the biggest
	connected pixel structure in the image. Fills this structure in white, reducing the rest to black.
	"""
	img = inp_img.copy()  # Copy the image, leaving the original untouched
	height, width = img.shape[:2]

	max_area = 0
	seed_point = (None, None)

	if scan_tl is None:
		scan_tl = [0, 0]

	if scan_br is None:
		scan_br = [width, height]

	# Loop through the image
	for x in range(scan_tl[0], scan_br[0]):
		for y in range(scan_tl[1], scan_br[1]):
			# Only operate on light or white squares
			if img.item(y, x) == 255 and x < width and y < height:  # Note that .item() appears to take input as y, x
				area = cv.floodFill(img, None, (x, y), 64)
				if area[0] > max_area:  # Gets the maximum bound area which should be the grid
					max_area = area[0]
					seed_point = (x, y)

	# Colour everything grey (compensates for features outside of our middle scanning range
	for x in range(width):
		for y in range(height):
			if img.item(y, x) == 255 and x < width and y < height:
				cv.floodFill(img, None, (x, y), 64)

	mask = np.zeros((height + 2, width + 2), np.uint8)  # Mask that is 2 pixels bigger than the image

	# Highlight the main feature
	if all([p is not None for p in seed_point]):
		cv.floodFill(img, mask, seed_point, 255)

	top, bottom, left, right = height, 0, width, 0

	for x in range(width):
		for y in range(height):
			if img.item(y, x) == 64:  # Hide anything that isn't the main feature
				cv.floodFill(img, mask, (x, y), 0)

			# Find the bounding parameters
			if img.item(y, x) == 255:
				top = y if y < top else top
				bottom = y if y > bottom else bottom
				left = x if x < left else left
				right = x if x > right else right

	bbox = [[left, top], [right, bottom]]
	return img, np.array(bbox, dtype='float32'), seed_point

def extract_digit(img, rect, size):
	"""Extracts a digit (if one exists) from a Sudoku square."""

	digit = cut_from_rect(img, rect)  # Get the digit box from the whole square

	# Use fill feature finding to get the largest feature in middle of the box
	# Margin used to define an area in the middle we would expect to find a pixel belonging to the digit
	h, w = digit.shape[:2]
	margin = int(np.mean([h, w]) / 2.5)
	_, bbox, seed = find_largest_feature(digit, [margin, margin], [w - margin, h - margin])
	digit = cut_from_rect(digit, bbox)

	# Scale and pad the digit so that it fits a square of the digit size we're using for machine learning
	w = bbox[1][0] - bbox[0][0]
	h = bbox[1][1] - bbox[0][1]

	# Ignore any small bounding boxes
	if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
		return scale_and_centre(digit, size, 4)
	else:
		return np.zeros((size, size), np.uint8)

def get_digits(img, squares, size):
    """Extracts digits from their cells and builds an array"""
    digits = []    
    img = preprocess_grid(img.copy())
    
    for square in squares:        
        digits.append(extract_digit(img, square, size))        
    return digits

def show_digits(digits, colour=255):
	"""Shows list of 81 extracted digits in a grid format"""
	rows = []
	with_border = [cv.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv.BORDER_CONSTANT, None, colour) for img in digits]
	for i in range(9):
		row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
		rows.append(row)
	display_img(np.concatenate(rows))

digits = get_digits(resized, squares, 28)
#for digit in digits:
#    display_img(digit)
show_digits(digits)

#TODO bayesian optimization on link between blur and blurring parameters ?

"""
preprocessed = preprocess_grid(resized)
display_img(preprocessed, title="preprocessed")
#preprocessed = preprocess_img(cell) # create a preprocess digit ? or preprocess cropped instead ! less blur and eliminate lines

digit_rects = []
for square in squares:
    (x1, y1), (x2, y2) = square
    cell = preprocessed[y1:y2, x1:x2]
    cell_area = cell.shape[0]*cell.shape[1]    
    contours, hierarchy = cv.findContours(cell, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    digit_rect = None
    max_area = 0
    for cnt in contours:  
        x,y,w,h = cv.boundingRect(cnt)
        rect_area = w*h
        if rect_area < cell_area / 2 and rect_area > cell_area / 15: 
            if rect_area > max_area:
                max_area = rect_area
                digit_rect = (x,y,w,h)

    if digit_rect:        
        x,y,w,h = digit_rect
        aspect_ratio = float(w)/h       
        if aspect_ratio > 0.2 and aspect_ratio < 0.8:
            digit_rects.append((x+x1,y+y1,w,h))

digits = resized.copy()
for rect in digit_rects:
    x,y,w,h = rect    
    cv.rectangle(digits, (x,y), (x+w,y+h), (0,0,255), 2)

display_img(digits, title="digits")
"""





# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
#im_with_keypoints = cv.drawKeypoints(digit, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# Show keypoints
#cv.imshow("Keypoints", im_with_keypoints)
#cv.waitKey(0)

"""
contours, hierarchy = cv.findContours(preprocessed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours_img = img.copy()
cv.drawContours(contours_img, contours, -1, (0, 0, 255), 2)
cv.imshow('contours', contours_img)
cv.waitKey(0)

sorted_contours = sorted(contours, key=lambda c: cv.contourArea(c), reverse=True)
biggest = sorted_contours[0]

approx_img = img.copy()
cv.drawContours(approx_img, [biggest], -1, (0, 255, 0), 2)
cv.imshow('approx', approx_img)
cv.waitKey(0)

# 3. Find corners
import operator
c = biggest
# determine the most extreme points along the contour

top_left = tuple(sorted(biggest, key=lambda p: p[0][0] + p[0][1])[0][0])
top_right = tuple(sorted(biggest, key=lambda p: p[0][0] - p[0][1], reverse=True)[0][0])
bottom_left = tuple(sorted(biggest, key=lambda p: p[0][0] - p[0][1])[0][0])
bottom_right = tuple(sorted(biggest, key=lambda p: p[0][0] + p[0][1], reverse=True)[0][0])


corner_img = img.copy()
cv.circle(corner_img, top_left, 8, (0, 0, 255), -1)
cv.circle(corner_img, top_right, 8, (0, 255, 0), -1)
cv.circle(corner_img, bottom_left, 8, (255, 0, 0), -1)
cv.circle(corner_img, bottom_right, 8, (255, 255, 0), -1)

cv.imshow('corners', corner_img)
cv.waitKey(0)

# 4. Perspective transform

def warp_and_crop(img, img_corners):
    L = int(np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))) # could do a mean
    dest_corners = np.float32([
        [0, 0],
        [L - 1, 0],
        [0, L - 1],
        [L - 1, L - 1],
    ])

    M = cv.getPerspectiveTransform(img_corners, dest_corners)
    return cv.warpPerspective(img, M, (L, L))

corners = np.float32([
    top_left,
    top_right,
    bottom_left,
    bottom_right
])

warped_img = warp_and_crop(img, corners)

cv.imshow('warped', warped_img)
cv.waitKey(0)

# 5. Infer grid

def infer_grid(img):
	squares = []
	side = img.shape[:1]
	side = side[0] / 9
	for i in range(9):
		for j in range(9):
			p1 = (i * side, j * side)  # Top left corner of a bounding box
			p2 = ((i + 1) * side, (j + 1) * side)  # Bottom right corner of bounding box
			squares.append((p1, p2))
	return squares

rects = infer_grid(warped_img)

grid_img = warped_img.copy()
for rect in rects:
    grid_img = cv.rectangle(grid_img, tuple(int(x) for x in rect[0]), tuple(int(x) for x in rect[1]), (255, 0, 0))

cv.imshow('grid', grid_img)
cv.waitKey(0)
"""