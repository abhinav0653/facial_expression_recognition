import cv2
import glob

faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet2 = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet3 = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet4 = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotions

def detect_faces(files):
#    files = glob.glob("./myImages/*.jpg") #Get list of all images with emotion
    #print files
    files = [files]
    filenumber = 0
    for f in files:
        frame = cv2.imread(f) #Open image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert image to grayscale
	face=0
	face2=0
	face3=0
	face4=0
	for scale in [float(i)/10 for i in range(11, 15)]:
        	for neighbors in range(1,5):
			#scale=1.1
			#neighbors=10
        		#Detect face using 4 different classifiers
        		face = faceDet.detectMultiScale(gray, scaleFactor=scale, minNeighbors=neighbors, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        		face2 = faceDet2.detectMultiScale(gray, scaleFactor=scale, minNeighbors=neighbors, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        		face3 = faceDet3.detectMultiScale(gray, scaleFactor=scale, minNeighbors=neighbors, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        		face4 = faceDet4.detectMultiScale(gray, scaleFactor=scale, minNeighbors=neighbors, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
			if(len(face)==1 or len(face2)==1 or len(face3)==1 or len(face4)==1):
				break
		if(len(face)==1 or len(face2)==1 or len(face3)==1 or len(face4)==1):
				break

        facefeatures = ""
        #Go over detected faces, stop at first detected face, return empty if no face.
        if len(face) == 1:
           	facefeatures = face
        elif len(face2) == 1:
            	facefeatures == face2
        elif len(face3) == 1:
            	facefeatures = face3
        elif len(face4) == 1:
            	facefeatures = face4
        else:
            	facefeatures = ""

        #Cut and save face
        for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
            print "face found in file: %s" %f
            gray = gray[y:y+h, x:x+w] #Cut the frame to size

            try:
                out = cv2.resize(gray, (64, 64)) #Resize face so all images have same size
                fext = f.split(".")[-1]
                fname = f.split("/")[-1]
                fname = fname.split(".")[0]
                print fname+"_converted."+fext
                cv2.imwrite("./tImages/"+fname+"."+fext, out) #Write image
            except:
               pass #If error, pass file
        filenumber += 1 #Increment image number

