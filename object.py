import os
import re
import cv2 # opencv library
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt

col_frames = os.listdir(r'C:\Users\End User\PycharmProjects\cloudchamber\time frame 10 min')

# sort file names
col_frames.sort(key=lambda f: int(re.sub('\D', '', f)))

# empty list to store the frames
col_images=[]

for i in col_frames:
    # read the frames
    img = cv2.imread(r'C:\Users\End User\PycharmProjects\cloudchamber\time frame 10 min/'+i)
    # append the frames to the list
    col_images.append(img)

i = 13

for frame in [i, i+1]:
    plt.imshow(cv2.cvtColor(col_images[frame], cv2.COLOR_BGR2RGB))
    plt.title(r"C:\Users\End User\PycharmProjects\cloudchamber\time frame 10 min: "+str(frame))
    plt.show()

# convert the frames to grayscale
grayA = cv2.cvtColor(col_images[i], cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(col_images[i+1], cv2.COLOR_BGR2GRAY)

# plot the image after frame differencing
plt.imshow(cv2.absdiff(grayB, grayA), cmap = 'gray')
plt.show()
diff_image = cv2.absdiff(grayB, grayA)

# perform image thresholding
ret, thresh = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)

# plot image after thresholding
plt.imshow(thresh, cmap = 'gray')
plt.show()

# apply image dilation
kernel = np.ones((3,3),np.uint8)
dilated = cv2.dilate(thresh,kernel,iterations = 1)

# plot dilated image
plt.imshow(dilated, cmap = 'gray')
plt.show()

# plot vehicle detection zone
plt.imshow(dilated)
cv2.line(dilated, (0, 100),(720,100),(100, 0, 0))
plt.show()

# find contours
contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

valid_cntrs = []

for i,cntr in enumerate(contours):
    x,y,w,h = cv2.boundingRect(cntr)
    if (x <= 720) & (y >= 100) & (cv2.contourArea(cntr) >= 50):
        valid_cntrs.append(cntr)

# count of discovered contours
len(valid_cntrs)

dmy = col_images[13].copy()

cv2.drawContours(dmy, valid_cntrs, -1, (720,100,0), 2)
cv2.line(dmy, (0, 100),(720,100),(100, 255, 255))
plt.imshow(dmy)
plt.show()

# kernel for image dilation
kernel = np.ones((4, 4), np.uint8)

# font style
font = cv2.FONT_HERSHEY_SIMPLEX

# directory to save the ouput frames
pathIn =r'C:\Users\End User\PycharmProjects\cloudchamber\time frame/'

for i in range(len(col_images) - 1):

    # frame differencing
    grayA = cv2.cvtColor(col_images[i], cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(col_images[i + 1], cv2.COLOR_BGR2GRAY)
    diff_image = cv2.absdiff(grayB, grayA)

    # image thresholding
    ret, thresh = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)

    # image dilation
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # find contours
    contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # shortlist contours appearing in the detection zone
    valid_cntrs = []
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        if (x <= 720) & (y >= 100) & (cv2.contourArea(cntr) >= 50):
            if (y >= 200) & (cv2.contourArea(cntr) < 10):
                break
            valid_cntrs.append(cntr)


    # add contours to original frames
    dmy = col_images[i].copy()
    cv2.drawContours(dmy, valid_cntrs, -1, (720, 100, 0), 2)

    cv2.putText(dmy, "particles detected: " + str(len(valid_cntrs)), (55, 15), font, 0.6, (0, 180, 0), 2)
    cv2.line(dmy, (0, 100), (720, 100), (100, 255, 255))
    cv2.imwrite(pathIn + str(i) + '.png', dmy)

    i = 0

    # list for storing names of shapes
    for contour in contours:

        # here we are ignoring first counter because
        # findcontour function detects whole image as shape
        if i == 0:
            i = 1
            continue

        # cv2.approxPloyDP() function to approximate the shape
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)

        # using drawContours() function
        cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)

        # finding center point of shape
        M = cv2.moments(contour)
        if M['m00'] != 0.0:
            x = int(M['m10'] / M['m00'])
            y = int(M['m01'] / M['m00'])

        # putting shape name at center of each shape
        if len(approx) == 3:
            cv2.putText(img, 'alpha', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elif len(approx) == 4:
            cv2.putText(img, 'muon', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elif len(approx) == 5:
            cv2.putText(img, 'electron', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elif len(approx) == 6:
            cv2.putText(img, 'photoelectron', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        else:
            cv2.putText(img, 'circle', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# specify video name
pathOut = r'C:\Users\End User\PycharmProjects\cloudchamber\first_10_mins.mp4v'

# specify frames per second
fps = 10.0

frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
files.sort(key=lambda f: int(re.sub('\D', '', f)))

print(files)
out = None

for i in range(len(files)):
    filename = pathIn + files[i]

    # read frames
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)

    # inserting the frames into an image array
    # frame_array.append(img)
    if out is None:
        out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    out.write(img)

# print(frame_array)
# for i in range(len(frame_array)):
    # writing to a image array
    # out.write(frame_array[i])

out.release()


