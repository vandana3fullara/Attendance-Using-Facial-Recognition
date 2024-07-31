# Import The Necessary Packages
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2

#Initialize 'currentname' To Trigger Only When A New Person Is Identified
currentname = "Unknown"

#Determine Faces From encodings.pickle File Model Created From train_model.py
encodingsP = "encodings.pickle"

# Load The Known Faces And Embeddings Along With OpenCV's Haarcascade For Face Detection
print("Loading Encodings + Face detector")
data = pickle.loads(open(encodingsP, "rb").read())

# Initialize The Video Stream And Allow The Camera Sensor To Warm Up
# Set The Ser To The Following
# src = 0 : For The Build In Single Web Cam, Could Be Your Laptop Webcam
# src = 2 : I Had To Set It To 2 Inorder To Use The Usb Webcam Attached To My Laptop 
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# Start The Frames Per Second Counter
fps = FPS().start()

# Loop Over Frames From The Video File Stream 
while True:
	# Grab The Frame From The Threaded Video Stream And Resize It
	# To 500px (To Speedup Processing)
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	# Detect The Face Boxes
	boxes = face_recognition.face_locations(frame)
	# Compute The Facial Embeddings For Each Face Bounding Box
	encodings = face_recognition.face_encodings(frame, boxes)
	names = []

	# Loop Over The Facial Embeddings
	for encoding in encodings:
		# Attempt To Match Each Face In The Input Image To Our Known Encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown" # If Face Is Not Recognized, Then Print Unknown

		# Check To See If We Have Found A Match
		if True in matches:
			# Find The Index Of All Matched Faces And Then Initialize A 
			# Dictionary To Count The Total Number Of Times Each Face Was Matched 
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# Loop Over The Matched Indexes And Maintain A Count For Each Recognized Face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# Determined The Recognized Face With The Largest Number Of Votes (Note: In The Event Of An Unlikely Tie Python Will
			# Select First Entry In The Dictionary)
			name = max(counts, key=counts.get)

			#If Someone In Your Dataset Is Identified, Print Their Name On The Screen
			if currentname != name:
				currentname = name
				print(currentname)

		# Update The List Of Names
		names.append(name)

	# Loop Over The Recognized Faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# Draw The Predicted Face Name On The Image - Color Is In Bgr
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 225), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			.8, (0, 255, 255), 2)

	# Display The Image To Our Screen 
	cv2.imshow("Facial Recognition Is Running", frame)
	key = cv2.waitKey(1) & 0xFF

	# Quit When 'q' Key Is Pressed
	if key == ord("q"):
		break

	# Update The Frames Per Second Counter
	fps.update()

# Stop The Timer And Display Frames Per Second Information 
fps.stop()
print("Elasped Time: {:.2f}".format(fps.elapsed()))
print("Approx Frames Per Second: {:.2f}".format(fps.fps()))

# Do A Bit Of Cleanup
cv2.destroyAllWindows()
vs.stop()
