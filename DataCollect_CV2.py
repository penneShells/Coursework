import cv2

def faceCollect(video, type, folder, neighbours):
    #Load the video using cv2 and assign the frame to the variable image
    vidCap = cv2.VideoCapture(video)
    success, image = vidCap.read()
    count = 0
    #Load the Haar cascade to detect faces
    faceDetector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

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
            cv2.imwrite(f"datasets/256_cv2/{folder}/frame%d.jpg" % counter, resized)
        count += 1
        success, image = vidCap.read()
    print(f"Data collected from {video}")


#Big function to run faceCollect on all videos.
#Will probably end up automating the calling of this with iteration but there are bigger fish to fry right now
#Also the dummy arguments are required for use in the GUI so just ignore them
def collectAll(dummy, dummy2):
    print("Collecting data - This may take a while")

    faceCollect("videos/video_archie.mp4", 1, "archie", 8)
    faceCollect("videos/video_archie_2.mp4", 2, "archie", 8)
    faceCollect("videos/video_archie_3.mp4", 3, "archie", 8)
    faceCollect("videos/video_archie_4.mp4", 4, "archie", 8)

    faceCollect("videos/video_alfie.mp4", 1, "alfie", 4)
    faceCollect("videos/video_alfie_2.mp4", 2, "alfie", 4)
    faceCollect("videos/video_alfie_3.mp4", 3, "alfie", 7)

    faceCollect("videos/video_keiram.mp4", 1, "keiran", 4)
    faceCollect("videos/video_keiran_2.mp4", 2, "keiran", 4)

    faceCollect("videos/video_josh.mp4", 1, "josh", 4)

    faceCollect("videos/video_ethan_1.mp4", 1, "ethan", 2)

    print("Data collection Achieved \n")
    print("You may wish to check the generated images for irregularities")