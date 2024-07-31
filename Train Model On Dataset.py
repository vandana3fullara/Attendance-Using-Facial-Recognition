# Import The Necessary Packages
from imutils import paths
import face_recognition
import pickle
import cv2
import os

# Our Images Are Located In The Dataset Folder 
print("Start Processing Faces")
imagePaths = list(paths.list_images("dataset"))

# Initialize The List Of Known Encodings And Known Names 
knownEncodings = []
knownNames = []

# Loop Over The Image Paths
for (i, imagePath) in enumerate(imagePaths):
	# Extract The Person Name From The Image Path
	print("Processing Image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# Load The Input Image And Convert It From Rgb (OpenCv Ordering) To Dlib Ordering (Rgb)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# Detect The (x, y) - Coordinates Of The Bounding Boxes
	# Corresponding To Each Face In The Input Image
	boxes = face_recognition.face_locations(rgb,
		model="hog")

	# Compute The Facial Embedding For The Face
	encodings = face_recognition.face_encodings(rgb, boxes)

	# Loop Over The Encodings
	for encoding in encodings:
		# Add Each Encoding + Name To Our Set Of Known Names And Encodings
		knownEncodings.append(encoding)
		knownNames.append(name)

# Dump The Facial Encodings + Names To Disk 
print("Serializing Encodings")
data = {"encodings": knownEncodings, "names": knownNames}
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()
