from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from imutils.video import VideoStream
#from imutils import face_utils

import argparse
import facenet
import imutils
#import os
#import sys
#import math
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
import dlib
from sklearn.svm import SVC
from scipy.spatial import distance
from imutils import face_utils


predictor = dlib.shape_predictor('Models/shape_predictor_68_face_landmarks.dat')
#predictor = dlib.shape_predictor('Models/shape_predictor_5_face_landmarks.dat')
eye_cascade = cv2.CascadeClassifier('../Models/haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('../Models/haarcascade_frontalface_default.xml')
left_eye_indices = list(range(36, 42))
right_eye_indices = list(range(42, 48))
threshold = 83
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
def detect_landmarks(frame, face):
    landmarks = predictor(frame, face)
    landmark_points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmark_points.append((x, y))
    return landmark_points

def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A + B) / (2.0 * C)
    return ear_aspect_ratio

def detect_blink(frame, shape):
    left_eye = [36, 37, 38, 39, 40, 41]  # Indices of landmarks for the left eye
    right_eye = [42, 43, 44, 45, 46, 47]  # Indices of landmarks for the right eye

    # Extract landmark coordinates for left and right eyes
    left_eye_landmarks = [shape[i] for i in left_eye]
    right_eye_landmarks = [shape[i] for i in right_eye]

    # Calculate EAR (Eye Aspect Ratio) for both eyes
    left_ear = calculate_EAR(left_eye_landmarks)
    right_ear = calculate_EAR(right_eye_landmarks)

    # Calculate the average EAR of both eyes
    distance = (left_ear + right_ear) / 2
    print("ear:" ,distance)

    for i in range(len(left_eye) - 1):
            cv2.line(frame, shape[left_eye[i]], shape[left_eye[i + 1]], (0, 0, 255), 1)
            cv2.line(frame, shape[right_eye[i]], shape[right_eye[i + 1]], (0, 0, 255), 1)
        # Connect the last point to the first point to complete the contour
    cv2.line(frame, shape[left_eye[-1]], shape[left_eye[0]], (0, 0, 255), 1)
    cv2.line(frame, shape[right_eye[-1]], shape[right_eye[0]], (0, 0, 255), 1)
    # Set a threshold to determine if the eyes are blinked
    EAR_THRESH = 0.2

    # Check if the EAR is below the threshold
    if distance < EAR_THRESH:
        return True  # The eyes are blinked
    else:
        return False 

# Sửa hàm smile
def smile(mouth):
    A = distance.euclidean(mouth[3], mouth[9])
    B = distance.euclidean(mouth[2], mouth[10])
    C = distance.euclidean(mouth[4], mouth[8])
    avg = (A+B+C)/3
    D = distance.euclidean(mouth[0], mouth[6])
    mar=avg/D
    return mar

# Sửa hàm detect_smile
def detect_smile(frame, shape): 
    # Tính tỉ lệ nụ cười
    mouth= shape[mStart:mEnd]
    mar = smile(mouth)
    print("mar: " ,mar )
    # Vẽ đường biên miệng
    mouthContour = shape[48:60]
    for i in range(len(mouth) - 1):
        cv2.line(frame, mouth[i], mouth[i+1], (0, 255, 0), 1)

    # Kết nối điểm cuối với điểm đầu để vẽ biên miệng hoàn chỉnh
    cv2.line(frame, mouth[-1], mouth[0], (0, 255, 0), 1)
    #mouthHull = cv2.convexHull(shape[48:60])
    #cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
    
    # Xác định nụ cười dựa trên ngưỡng smile_threshold
    if  mar <= .3 or mar > 0.45:
        return True
    else:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path of the video you want to test on.', default=0)
    args = parser.parse_args()

    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160
    CLASSIFIER_PATH = 'Models/facemodel.pkl'
    VIDEO_PATH = args.path
    FACENET_MODEL_PATH = 'Models/20180402-114759.pb'

    # Load The Custom Classifier
    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    print("Custom Classifier, Successfully loaded")

    with tf.Graph().as_default():

        # Cai dat GPU neu co
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)

            # Get input and output tensors
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")

            people_detected = set()
            person_detected = collections.Counter()

            cap  = VideoStream(src=0).start()

            while (True):
                frame = cap.read()
                frame = imutils.resize(frame, width=600)
                frame = cv2.flip(frame, 1)

                bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

                faces_found = bounding_boxes.shape[0]
                try:
                    if faces_found > 10:
                        cv2.putText(frame, "Only one face", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (255, 255, 255), thickness=1, lineType=2)
                    elif faces_found > 0:
                        det = bounding_boxes[:, 0:4]
                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        for i in range(faces_found):
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]
                             # Phát hiện landmark 68 điểm
                            face_rect = dlib.rectangle(bb[i][0], bb[i][1], bb[i][2], bb[i][3])
                            landmark_points = detect_landmarks(frame, face_rect)
                            # Vẽ các điểm landmark
                            for (x, y) in landmark_points:
                                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1) 
                               
                            
                            print(bb[i][3]-bb[i][1])
                            print(frame.shape[0])
                            print((bb[i][3]-bb[i][1])/frame.shape[0])
                            if (bb[i][3]-bb[i][1])/frame.shape[0]>0.25:
                                cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                                scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                    interpolation=cv2.INTER_CUBIC)
                                scaled = facenet.prewhiten(scaled)
                                scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                                emb_array = sess.run(embeddings, feed_dict=feed_dict)

                                predictions = model.predict_proba(emb_array)
                                best_class_indices = np.argmax(predictions, axis=1)
                                best_class_probabilities = predictions[
                                    np.arange(len(best_class_indices)), best_class_indices]
                                best_name = class_names[best_class_indices[0]]
                                print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))



                                if best_class_probabilities > 0.8:
                                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                    text_x = bb[i][0]
                                    text_y = bb[i][3] + 20

                                    name = class_names[best_class_indices[0]]
                                    cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 255, 255), thickness=1, lineType=2)
                                    cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17),
                                                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 255, 255), thickness=1, lineType=2)
                                    person_detected[best_name] += 1
                                    
                                else:
                                    name = "Unknown"
                            if detect_smile(frame,landmark_points):
                                
                                cv2.putText(frame, "Smiling", (text_x, text_y + 51), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 255, 255), thickness=1, lineType=2)  
                                print("smiling") 
                            if detect_blink(frame,landmark_points):
                
                                cv2.putText(frame, "mat nham", (text_x, text_y + 34), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 255, 255), thickness=1, lineType=2)
                                print("Mắt nhắm")

                            #mouth_landmarks = landmark_points[48:60]  # Assuming these are the landmarks for the mouth
                                                   

                except:
                    pass
                #if detect_blink(frame):
                    #print("Mắt nhắm")
                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()


main()