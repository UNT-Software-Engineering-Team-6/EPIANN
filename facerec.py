# facerec.py
# This is the main File
import cv2
import numpy
import os

size = 2
haar_cascade = cv2.CascadeClassifier(r'C:\Users\hp\Documents\GitHub\Face-Recognition-For-Criminal-Detection-GUi\face_cascade.xml')




# Part 1: Create fisherRecognizer
def train_model():
    model = cv2.face.LBPHFaceRecognizer_create()
    fn_dir = 'face_samples'

    print('Training...')

    (images, lables, names, id) = ([], [], {}, 0)

    for (subdirs, dirs, files) in os.walk(fn_dir):
        # Loop through each folder named after the subject in the photos
        for subdir in dirs:
            names[id] = subdir
            subjectpath = os.path.join(fn_dir, subdir)
            # Loop through each photo in the folder
            for filename in os.listdir(subjectpath):
                # Skip non-image formates
                f_name, f_extension = os.path.splitext(filename)
                if(f_extension.lower() not in ['.png','.jpg','.jpeg','.gif','.pgm']):
                    print("Skipping "+filename+", wrong file type")
                    continue
                path = subjectpath + '/' + filename
                lable = id
                # Add to training data
                images.append(cv2.imread(path, 0))
                lables.append(int(lable))
            id += 1

    # Create a Numpy array from the two lists above
    (images, lables) = [numpy.array(lis) for lis in [images, lables]]
    # OpenCV trains a model from the images
    model.train(images, lables)

    return (model, names)


# Part 2: Use fisherRecognizer on camera stream
def detect_faces(gray_frame):
    global size, haar_cascade

    # Resize to speed up detection (optinal, change size above)
    mini_frame = cv2.resize(gray_frame, (int(gray_frame.shape[1] / size), int(gray_frame.shape[0] / size)))

    # Detect faces and loop through each one
    faces = haar_cascade.detectMultiScale(mini_frame)
    return faces

#code to recognize the face 
def recognize_face(model, frame, gray_frame, face_coords, names):
    (img_width, img_height) = (112, 92)
    recognized = []
    recog_names = []

    for i in range(len(face_coords)):
        face_i = face_coords[i]

        # Coordinates of face after scaling back by `size`
        (x, y, w, h) = [v * size for v in face_i]
        face = gray_frame[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (img_width, img_height))

        # Try to recognize the face
        (prediction, confidence) = model.predict(face_resize)

        # print(prediction, confidence)
        if (confidence<95 and names[prediction] not in recog_names):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            recog_names.append(names[prediction])
            recognized.append((names[prediction].capitalize(), confidence))
        elif (confidence >= 95):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return (frame, recognized)

#train_model()

from handler import retrieveData
import numpy as np
from playsound import playsound
from mtcnn import MTCNN 
import pygame

pygame.init()

def play_alert_sound():
    pygame.mixer.music.load("alert.mp3")
    pygame.mixer.music.play()

    
size = 2
detector = MTCNN()

def train_model2():
    
    model = cv2.face.LBPHFaceRecognizer_create()

    fn_dir = 'face_samples'

    print('Training...')

    (images, labels, names, id) = ([], [], {}, 0)

    for (subdirs, dirs, files) in os.walk(fn_dir):
        # Loop through each folder named after the subject in the photos
        for subdir in dirs:
            names[id] = subdir
            subjectpath = os.path.join(fn_dir, subdir)
            # Loop through each photo in the folder
            for filename in os.listdir(subjectpath):
                # Skip non-image formats
                f_name, f_extension = os.path.splitext(filename)
                if f_extension.lower() not in ['.png', '.jpg', '.jpeg', '.gif', '.pgm']:
                    print("Skipping "+filename+", wrong file type")
                    continue
                path = os.path.join(subjectpath, filename)
                label = id
                # Add to training data
                images.append(cv2.imread(path, 0))
                labels.append(int(label))
            id += 1

    # Create numpy arrays from the lists
    (images, labels) = [np.array(lst) for lst in [images, labels]]
    # Train the model
    model.train(images, labels)

    return (model, names)

# Part 2: Use MTCNN for face detection
def detect_faces2(gray_frame):
    bgr_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)  # Convert to BGR format
    faces = detector.detect_faces(bgr_frame)
    face_coords = [[face['box'][0], face['box'][1], face['box'][2], face['box'][3]] for face in faces]
    return face_coords

    

#train_model()# Part 3: Recognize faces
# Part 3: Recognize faces
def recognize_face2(model, frame, gray_frame, face_coords, names):
    (img_width, img_height) = (112, 92)
    recognized = []
    recog_names = []

    for (x, y, w, h) in face_coords:
        face = gray_frame[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (img_width, img_height))

        # Recognize the face
        prediction, confidence = model.predict(face_resize)

        if confidence < 95:
            name = names.get(prediction, "Unknown") # Get the name from the dictionary or default to "Unknown"
            _, crim_data = retrieveData(name)
            if "Crimes" in crim_data:
                crimes = int(crim_data["Crimes"])
                if crimes == 0:
                    # Green rectangle if no crimes committed
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    if name not in recog_names:
                        recog_names.append(name)
                        recognized.append((name.capitalize(), confidence))
                        text_width, _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                        font_scale = min(w / text_width, 1)
                        cv2.putText(frame, name, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2, cv2.LINE_AA)
                    play_safe_sound()
                else:
                    # Red rectangle if crimes committed
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    if name not in recog_names:
                        recog_names.append(name)
                        recognized.append((name.capitalize(), confidence))
                        text_width, _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                        font_scale = min(w / text_width, 1)
                        cv2.putText(frame, name, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2, cv2.LINE_AA)
                    play_alert_sound()
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 5)  # Orange rectangle when face is not recognized
            cv2.putText(frame, "Not Identified", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2, cv2.LINE_AA)

    # frame = cv2.flip(frame, 1)
    return (frame, recognized)

# Train the model
(model, names) = train_model()
