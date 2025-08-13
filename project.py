import numpy as np 
import cv2
from keras.preprocessing import image
from keras.models import model_from_json 

# Load face detection model
face_cascade = cv2.CascadeClassifier('C:\\Users\\reddy\\OneDrive\\Desktop\\SCT Project\\haarcascade_frontalface_alt.xml')

# Load facial expression recognition model
model = model_from_json(open(r"C:\Users\reddy\OneDrive\Desktop\SCT Project\facial_expression_model_structure.json", "r").read()) 
model.load_weights(r'C:\Users\reddy\OneDrive\Desktop\SCT Project\facial_expression_model_weights.h5')

# Initialize video capture from the default camera
video_capture = cv2.VideoCapture(0)

# Define emotions
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# Check if the camera is opened successfully
if not video_capture.isOpened():
    print("Error: Could not open camera.")
else:
    # Read frames from the camera
    while True:
        ret, img = video_capture.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces: 
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            detected_face = img[int(y):int(y+h), int(x):int(x+w)] 
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) 
            detected_face = cv2.resize(detected_face, (48, 48))
            img_pixels = image.img_to_array(detected_face) 
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            img_pixels /= 255
            predictions = model.predict(img_pixels)
            max_index = np.argmax(predictions[0])
            emotion = emotions[max_index]
            cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow('img',img)
        
        # Display the captured frame
        cv2.imshow('Camera', img)
        
        # Wait for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

# Release the camera and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
