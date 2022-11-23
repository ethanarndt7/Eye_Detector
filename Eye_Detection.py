import cv2

# Eye detection data from haarcascade algorithm
smile_data = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

# Captures the video from webcam
video = cv2.VideoCapture(0)

while True:
    # Reads the frame
    successful_frame_read, frame = video.read()

    # Converts frames to gray scale
    gs_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gets face coordinates
    coordinates = smile_data.detectMultiScale(gs_frame)

    # Draws Rectangles around eyes
    for (x, y, w, h) in coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0 , 255), 2)

    # Show video
    cv2.imshow('Eye Detection (press Q to quit)', frame)
    key = cv2.waitKey(1)    # makes video real time

    # Sets Q as stopping key
    if key==81 or key==113:
        break