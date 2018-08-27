import argparse
import imutils
import datetime
import time
import cv2 as cv

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", 
	help="path to the video file")
ap.add_argument("-a", "--area", type=int, default=500,
	help="minimum area size");
args = vars(ap.parse_args())

if args.get("vedio", None) is None:
	camera = cv.VideoCapture(0)
	time.sleep(0.25)

else:
	camera = cv.VideoCapture(args["video"])

firstFrame = None

while True:
	
	# grab the current frame and initialize the unoccupied to display
	(grabbed, frame) = camera.read() 
	text = "Nobody"

	if not grabbed:
		break

	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=500)
	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	gray = cv.GaussianBlur(gray, (21,21), 0)

	if firstFrame is None:
		firstFrame = gray
		continue

	# find the difference and apply thresholding to the frame 
	frameDelta = cv.absdiff(firstFrame, gray)
	thresh = cv.threshold(frameDelta, 25, 255, cv.THRESH_BINARY)[1]

	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	thresh = cv.dilate(thresh, None, iterations=2)
	(_, counts, _) = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


	for c in counts:

		if cv.contourArea(c) < args["area"]:
			continue

		# compute the bounding box for the contour, draw it on the frame change the text
		(x, y, w, h) = cv.boundingRect(c)
		cv.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
		text = "Someone's here"

	# draw the text and timestamp on the frame
	cv.putText(frame, "Room status: {}".format(text), (10, 20),
	cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
	cv.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
		(10, frame.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255), 1)
	

	cv.imshow("Security feed", frame)
	cv.imshow("Thresh", thresh)
	cv.imshow("Frame Delta", frameDelta)
	key = cv.waitKey(1) & 0xFF

	if key == ord("q"):
		break

camera.release()
cv.destroyAllWindows()




