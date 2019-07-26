#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int16MultiArray
from std_msgs.msg import Float32MultiArray

import cv2
from cv_bridge import CvBridge, CvBridgeError

import sys
sys.path.append("")

# from __future__ import division, print_function
import os
import time
from decimal import Decimal, ROUND_HALF_UP

import warnings
import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.layers import Input
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.layers import Conv2D
from keras.regularizers import l2
from keras.layers.core import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras.utils import layer_utils
from keras.preprocessing import image
# from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session

from places_utils import preprocess_input 
from vgg16_places_365 import VGG16_Places365

#GPU usage limit##############################
config = tf.ConfigProto(
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=0.20,
            visible_device_list="0",
            allow_growth=True
    )
)
set_session(tf.Session(config=config))
##############################################

# load the class label
file_name = '/home/amsl/ros_catkin_ws/src/recognition/vgg16_places_ros/labels/categories_places365.txt'
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

#input raw video from usb_cam
model = VGG16_Places365(weights='places')
predictions_to_return = 3

input_shape = (224, 224)

class Places365Camera(object):
    def __init__(self, class_names, model, input_shape):
        self.class_names = class_names
        self.model = model
        self.input_shape = input_shape
        # self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.ImageCallback)
        self.image_sub = rospy.Subscriber("/camera/rgb/resized_image", Image, self.ImageCallback)
        # self.image_sub = rospy.Subscriber("/zed0/left/image_rect_color", Image, self.ImageCallback)
        self.pub_labels = rospy.Publisher("/place_category", Int16MultiArray, queue_size=10)
        self.pub_prob = rospy.Publisher("/place_probability", Float32MultiArray, queue_size=10)

    def ImageCallback(self, image_msg):
        try:
            self.cv_image = CvBridge().imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeerror as e:
            print (e)

    def run(self):
        start = time.time()
        image = cv2.resize(self.cv_image, (self.input_shape[0], self.input_shape[1]))
        tmp_inp = np.array(image, dtype=np.uint8)
        tmp_inp = np.expand_dims(tmp_inp, 0)
        preds = model.predict(tmp_inp)[0]

        top_preds = np.argsort(preds)[::-1][0:predictions_to_return]

        # print('\n' + '--PREDICTED SCENE CATEGORIES:')
        # output the prediction
        pub_num = Int16MultiArray()
        prob = Float32MultiArray()

        for i in range(0, 3):
            # if preds[top_preds[i]] > 0.1:
            print classes[top_preds[i]],
            prob.data.append(Decimal(str(preds[top_preds[i]])).quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP))
            pub_num.data.append(top_preds[i])
        print ''
        
        # print 'processing time:', time.time() - start

        pub_num.layout.data_offset = 5
        self.pub_labels.publish(pub_num)
        self.pub_prob.publish(prob)

    def main(self):
        rospy.init_node("places365_ros")
        # rate = rospy.Rate(20)
        rate = rospy.Rate(5)

        while not rospy.is_shutdown():
            self.run()
            rate.sleep()

camera = Places365Camera(classes, model, input_shape)
camera.main()
