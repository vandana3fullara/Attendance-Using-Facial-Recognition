import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray

name = 'Vandana Fullara' 

cam = PiCamera()
cam.resolution = (512, 304)
cam.framerate = 10
rawCapture = PiRGBArray(cam, size=(512, 304))
    
img_counter = 0

while True:
    for frame in cam.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        cv2.imshow("Press Space To Take A Photo", image)
        rawCapture.truncate(0)
    
        k = cv2.waitKey(1)
        rawCapture.truncate(0)
        if k%256 == 27: # Press Escape Key
            break
        elif k%256 == 32:
            # Press Space Key
            img_name = "dataset/"+ name +"/image_{}.jpg".format(img_counter)
            cv2.imwrite(img_name, image)
            print("{} Written!".format(img_name))
            img_counter += 1
            
    if k%256 == 27:
        print("Escape Hit Closing")
        break

cv2.destroyAllWindows()
