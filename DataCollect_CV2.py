import cv2
import os

def faceCollect(video, type, folder, neighbours):
    #Load the video using cv2 and assign the frame to the variable image
    vidCap = cv2.VideoCapture(video)
    success, image = vidCap.read()
    count = 0
    #Load the Haar cascade to detect faces
    faceDetector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    print(success)
    while success:
        #Convert to gray to make facial detection easier
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #Detect faces and store their coordinates in an array
        faces = faceDetector.detectMultiScale(
            gray,
            minNeighbors=neighbours,
            minSize=(192, 192) #???
        )

        #Crop the image to just contain the detected face and resize to the desired resolution
        if len(faces) != 0:
            for (x, y, w, h) in faces:
                cropped = image[y: y + h, x: x + w]

            resized = cv2.resize(cropped, (256, 256))
            #this is absolutely terrible
            counter = count + ((type - 1) * 1000)

            #Save to a dataset folder
            cv2.imwrite(f"datasets/256_cv2_test/{folder}/frame%d.jpg" % counter, resized)
        count += 1
        success, image = vidCap.read()
    print(f"Data collected from {video}")


#Big function to run faceCollect on all videos.
#Dummy arguments are required for use in the GUI so just ignore them
def collectAll(dummy, dummy2):
    print("Collecting data - This may take a while")
    videos = [video for video in os.listdir('videos')]
    for video in videos:
        print(video)
        name = video.split("_")[1]
        faceCollect(f"videos/{video}", int(video[len(video) - 1]), name, 4)

    print("Data collection Achieved \n")
    print("You may wish to check the generated images for irregularities")


if __name__ == "__main__":
    collectAll(None, None)
