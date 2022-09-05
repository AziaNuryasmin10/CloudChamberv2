import numpy as np
import cv2
from skimage import data, filters
#from contours import join_contours
from past.builtins import xrange

# Open Video
cap = cv2.VideoCapture(r'C:\Users\End User\PycharmProjects\cloudchamber\video\first one min.mp4')

# setup output video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(r'C:\Users\End User\PycharmProjects\cloudchamber\video\output.avi', fourcc, 20.0, size)
out2 = cv2.VideoWriter(r'C:\Users\End User\PycharmProjects\cloudchamber\video\output2.avi', fourcc, 20.0, size)

# # edit your video size here, to adjust the performance
# largura = frame.shape[1]
# altura = frame.shape[0]
# lamenor = int(frame.shape[1] / 5)
# altmenor = int(frame.shape[0] / 5)
# frame = cv2.resize(frame, (lamenor, altmenor))

# Randomly select 25 frames
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)

# Store selected frames in an array
frames = []
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    # print(ret)
    frames.append(frame)

# Calculate the median along the time axis
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

# Display median frame
# cv2.imshow('frame', medianFrame)
# cv2.waitKey(0)

# Reset frame number to 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Convert background to grayscale
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)


def find_if_close(cnt1, cnt2):
    row1, row2 = cnt1.shape[0], cnt2.shape[0]
    for i in xrange(row1):
        for j in xrange(row2):
            dist = np.linalg.norm(cnt1[i] - cnt2[j])
            if abs(dist) < 100:
                return True
            elif i == row1 - 1 and j == row2 - 1:
                return False


def join_contours(contours):
    LENGTH = len(contours)
    status = np.zeros((LENGTH, 1))

    for i, cnt1 in enumerate(contours):
        x = i
        if i != LENGTH - 1:
            for j, cnt2 in enumerate(contours[i + 1:]):
                x = x + 1
                dist = find_if_close(cnt1, cnt2)
                if dist == True:
                    val = min(status[i], status[x])
                    status[x] = status[i] = val
                else:
                    if status[x] == status[i]:
                        status[x] = i + 1

    unified = []
    maximum = int(status.max()) + 1
    for i in xrange(maximum):
        pos = np.where(status == i)[0]
        if pos.size != 0:
            cont = np.vstack(contours[i] for i in pos)
            hull = cv2.convexHull(cont)
            unified.append(hull)

    return unified


# Loop over all frames
ret = True
while (ret):

    # Read frame
    ret, frame = cap.read()

    if ret == True:
        # Convert current frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculate absolute difference of current frame and
        # the median frame
        dframe = cv2.absdiff(frame, grayMedianFrame)
        # Treshold to binarize
        th, dframe = cv2.threshold(dframe, 70, 255, cv2.THRESH_BINARY)

        backtorgb = cv2.cvtColor(dframe, cv2.COLOR_GRAY2RGB)
        backtorgb2 = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        contours, hierarchy = cv2.findContours(
            dframe, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours and len(contours) > 250:
            contours = join_contours(contours)

        font = cv2.FONT_HERSHEY_COMPLEX

        for cnt in contours:
            if cv2.contourArea(cnt) > 200:
                image = cv2.drawContours(backtorgb, cnt, -1, (0, 255, 0), 2)
                image2 = cv2.drawContours(backtorgb2, cnt, -1, (0, 255, 0), 2)

                approx = cv2.approxPolyDP(
                    cnt, 0.009 * cv2.arcLength(cnt, True), True)

                n = approx.ravel()
                i = 0

                for j in n:
                    if (i % 2 == 0):
                        x = n[i]
                        y = n[i + 1]

                        string = str(x) + " " + str(y)

                        if (i == 0):
                            x_start = int(x)
                            y_start = int(y)

                        if (i != 0):
                            x_end = int(x)
                            y_end = int(y)
                    i = i + 1

                arrow = cv2.arrowedLine(
                    image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 3, tipLength=0.5)
                arrow2 = cv2.arrowedLine(
                    image2, (x_start, y_start), (x_end, y_end), (0, 0, 255), 3, tipLength=0.5)

        # Display & write image
        out.write(arrow)
        out2.write(arrow2)

        # Display image
        # cv2.imshow('image', arrow)
        # cv2.imshow('image2', arrow2)
        cv2.waitKey(10)

    else:
        break

# Release video object
cap.release()

# Destroy all windows
cv2.destroyAllWindows()
