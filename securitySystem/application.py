# USAGE
# python3 application.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages

import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print ("Time Taken to Load Libraries")
start_time = time.time()
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import h5py
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
from imutils.video import VideoStream
import cv2
import numpy as np
import argparse
import imutils
from PIL import Image
print ("Libraries Loaded")
print ("--- %s seconds ---" % (time.time() - start_time))

def buildFaceRecognitionModel():
    # To Train Model for the first time
    # FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
    # FRmodel.save_weights("weights/model.h5")
    # print("Saved model to disk")
    # load_weights_from_FaceNet(FRmodel)    
    print ("Time Taken to Load model")
    start_time = time.time()
    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    # print("Total Params:", FRmodel.count_params())
    FRmodel.load_weights("weights/model.h5")
    print("Loaded model from disk")
    print("--- %s seconds ---" % (time.time() - start_time))    
    return FRmodel   

def initialSetup():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=True,
        help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
        help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())
    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    # initialize the video stream and allow the cammera sensor to warmup
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    return args, net, vs

def encodingTrainingImages(FRmodel):
    print ("Time Taken to Create Image Encodings")
    start_time = time.time()
    database = {}
    database["Keshav"] = img_to_encoding("images/keshav.jpg", FRmodel) 
    database["LebronJames"] = img_to_encoding("images/lebron.png", FRmodel)
    database["LebronJames1"] = img_to_encoding("images/lebron1.jpg", FRmodel)
    database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
    # database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)  
    # database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
    # database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
    # database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
    # database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
    # database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
    # database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
    # database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
    # database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
    # database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
    # database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)
    print ("All the training images have been encoded")
    print ("--- %s seconds ---" % (time.time() - start_time))
    return database         

def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    ### START CODE HERE ### (≈ 4 lines)
    # Step 1: Compute the (encoding) distance between the anchor and the positive
    pos_dist = tf.square(anchor-positive)
    # Step 2: Compute the (encoding) distance between the anchor and the negative
    neg_dist = tf.square(anchor-negative)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.reduce_sum(pos_dist-neg_dist)+alpha
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss,0.))
    ### END CODE HERE ### 
    return loss    

def faceDetectionByFrame(args, net, vs, database, FRmodel):
    j=0
    change = True
    text = "Loading ..."    
    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=1000)
        j=j+1
        if (j == 25):
                j =  0
        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))
        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < args["confidence"]:
                continue
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")     
            # draw the bounding box of the face along with the associated
            # probability
            # text = " - {:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)	
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            
            # print ("Debugging")
            # print (j)
            # if j==12:
 
            # if change:
            #         text ="kesh"
            # else:
            #         text = "Bg"	
            # change = not change	
            crop_img = frame[startY:startY+h,startX:endX]
            #cv2.putText(frame, text1, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            crop_img1=cv2.resize(crop_img, (96,96), interpolation = cv2.INTER_AREA)
            arr = np.array(crop_img1)
            print (arr.shape)
            # Recognise Person's Face 
            min_dist, identity = who_is_it(arr, database, FRmodel)
            if min_dist > 0.7:
                # print("Not in the database : Unknown Person")
                text = "Unknown"
            else:
                # print ("it's " + str(identity) + ", the distance is " + str(min_dist)) 
                text = identity              
            #img = Image.fromarray(arr, 'RGB')
            #img.show()

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

def who_is_it(image_encoding, database, model):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.
    
    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras
    
    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """

    ### START CODE HERE ### 
    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding_directly() see example above. ## (≈ 1 line)
    encoding = img_to_encoding_directly(image_encoding, model)
    ## Step 2: Find the closest encoding ##
    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = np.linalg.norm(encoding-database[name])
        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist<min_dist:
            min_dist = dist
            identity = name
    ### END CODE HERE ###
    return min_dist, identity            

FRmodel = buildFaceRecognitionModel()
database = encodingTrainingImages(FRmodel)
args, net, vs = initialSetup()
faceDetectionByFrame(args, net, vs, database, FRmodel)

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()