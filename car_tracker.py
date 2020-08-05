import cv2

# load video with cars
#video = cv2.VideoCapture('videos/motorbike_dashcam.mp4')
video = cv2.VideoCapture('videos/tesla_dashcam.mp4')

# load pre-trained car classifier
car_classifier_file = 'classifiers/car_classifier.xml'
pedestrian_classifier_file = 'classifiers/haarcascade_fullbody.xml'

# create car and pedetrian classifier
car_tracker = cv2.CascadeClassifier(car_classifier_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_classifier_file)

while True:

    #Read the current frame
    (read_successful, frame) = video.read()

    if read_successful:
        #convert to grayscale (needed for haar cascade)
        #also faster to process
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    else:
        break

    #detect cars and pedestrians
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    # draw rects around detected cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # draw rects around detected pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    #Display frames with detected cars and pedestrians
    cv2.imshow('Video Car Detector', frame)

    #Stay on frame for 1 ms and continue
    cv2.waitKey(1)


print("No errors!")
