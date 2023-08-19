import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import playsound

model = load_model(r"C:\Users\yamin\OneDrive\Desktop\Emotion_Detection\model.h5")

video = cv2.VideoCapture(0)
Index = ["angry","disgust","fear","happy","neutral","sad","surprise"]
face_detector=cv2.CascadeClassifier(r'C:\Users\yamin\OneDrive\Desktop\Emotion_Detection\frontalface.xml')

def process_sound(val):
    path = r"C:\Users\yamin\OneDrive\Desktop\Emotion_Detection"

    if val == Index[0]:
        playsound.playsound(path+r"\audio\Angry.mp3")
    elif val == Index[1]:
        playsound.playsound(path+r"\audio\Disgust.mp3")
    elif val == Index[2]:
        playsound.playsound(path+r"\audio\Fear.mp3")
    elif val == Index[3]:
        playsound.playsound(path+r"\audio\Happy.mp3")
    elif val == Index[4]:
        playsound.playsound(path+r"\audio\Neutral.mp3")
    elif val == Index[5]:
        playsound.playsound(path+r"\audio\Sad.mp3")
    elif val == Index[6]:
        playsound.playsound(path+r"\audio\Surprise.mp3")

def live_stream():
    result = ""
    while True:
        ret,frame = video.read()
        frame = cv2.flip(frame, 1)
         
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_result = face_detector.detectMultiScale(gray, 1.1, 4)
        
        result = None
        for (x,y,w,h) in faces_result:
            frame1 = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            frame1 = cv2.resize(gray[y:y+h, x:x+w], (48,48))
            cv2.imwrite('image.jpg',frame1)
            img = image.load_img('image.jpg',target_size=(48,48),grayscale=True)
            x = image.img_to_array(img)
            x = np.expand_dims(x,axis = 0)
            k = model.predict(x)
            pred = np.argmax(k)
            a = np.argmax(k,axis=1)
            result = str(Index[a[0]])
            print()
            print("Detected class is: ",result)
            
            frame = cv2.putText(frame, result,(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),4)
        cv2.imshow('img',frame)
        if result != None:
            process_sound(result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

live_stream()

