#!/usr/bin/python3

import cv2
import sys
import numpy as np
import datetime
import os

def detectMotion(url, video_dest, time_threshold=3.0, size_threshold=0.01):
    cap = cv2.VideoCapture(url)
    if (cap.isOpened() == False):
        print('!!! Unable to open URL')
        return

    # retrieve FPS and calculate how long to wait between each frame to be display
    fps = cap.get(cv2.CAP_PROP_FPS)

    # create a background subtractor
    #backSub = cv2.createBackgroundSubtractorMOG2(
    #    varThreshold=25, detectShadows=True)
    backSub = cv2.createBackgroundSubtractorKNN()

    # Kernal for morphological operation
    kernel = np.ones((20, 20), np.uint8)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_timer = 0
    out = None

    # loop through each frame
    stream_ready = False
    while(True):
        # read one frame
        ret, frame = cap.read()
        if not ret:
            print("Stream not ready!")
            break;

        if not stream_ready:
            print("Stream is ready!")
            stream_ready = True

        # make the frame smaller so that we aren't using loads of cpu time.
        scale = 0.3
        dim = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
        scaled_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

        gray = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Apply our background subtraction
        fgMask = backSub.apply(gray)

        # Clean it up a bit
        fgMask = cv2.morphologyEx(fgMask,cv2.MORPH_CLOSE, kernel)
        fgMask = cv2.medianBlur(fgMask, 5)
        _, fgMask = cv2.threshold(fgMask, 127, 255, cv2.THRESH_BINARY)

        # Find the contours of the object inside the binary image
        contours, hierarchy = cv2.findContours(fgMask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        areas = [cv2.contourArea(c) for c in contours]

        # Find the total area of the countours
        total_area = 0
        for cnt in contours:
            hull = cv2.convexHull(cnt)
            total_area += cv2.contourArea(hull)

        #print(str(total_area))

        # Check whether we should start recording?
        size_threshold_area = size_threshold * frame_width * frame_height * scale

        #print(str(size_threshold_area))

        #print("AREA: " + str(area) + " THRESHOLD: " + str(size_threshold_area))
        if total_area > size_threshold_area :
            if not out:
                date = datetime.datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
                out = cv2.VideoWriter(os.path.join(video_dest, date + ".mkv"), cv2.VideoWriter_fourcc('X', '2', '6', '4'), fps, (frame_width, frame_height))

                print("Starting recording...")

            # reset the video timer
            video_timer = time_threshold

        # record video if we are recording
        if out:
            out.write(frame)

            # decrement our timer
            video_timer -= 1.0/fps
            if video_timer <= 0:
                out.release()
                out = None

                print("Finished recording...")

        # display frame
        #cv2.imshow('frame', scaled_frame)
        #cv2.imshow('FG Mask', fgMask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()

    #cv2.destroyAllWindows()

# Run as a script
if __name__ == "__main__":
    # Get command line arguments
    source = sys.argv[1]
    video_dest = sys.argv[2]

    # Detect motion!
    detectMotion(source, video_dest)
