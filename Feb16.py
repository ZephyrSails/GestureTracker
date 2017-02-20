import cv2
import numpy as np
import copy
from PIL import Image
import tensorflow as tf
import Feb14 as fb

import math

# parameters
cap_region_x_begin = 0.5    # start point/total width
cap_region_y_end = 0.8      # start point/total width
threshold = 60              #  BINARY threshold
blurValue = 41              # GaussianBlur parameter
bgSubThreshold = 16
counter = 0
# variables
isBgCaptured = 1   # bool, whether the background captured

save_path = './save_model.ckpt'

# Camera
camera = cv2.VideoCapture(0)
camera.set(10, 200)

xs = tf.placeholder(tf.float32, shape=([1,32, 32]))
x_image = tf.reshape(xs, [-1, 32, 32, 1])

keep_prob = tf.placeholder(tf.float32)
weights = {
            'W_conv1': fb.weight_variable([5, 5, 1, 32]),
            'W_conv2': fb.weight_variable([5, 5, 32, 64]),
            'W_fc1': fb.weight_variable([8 * 8 * 64, 1024]),
            'W_fc2': fb.weight_variable([1024, 3])

}
biases = {
            'b_conv1': fb.bias_variable([32]),
            'b_conv2': fb.bias_variable([64]),
            'b_fc1': fb.bias_variable([1024]),
            'b_fc2': fb.bias_variable([3])
}

prediction = fb.compute_net(x_image, weights, biases, keep_prob)

with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    # saver = tf.train.import_meta_graph('./save_model.ckpt.meta')
    # saver.restore(sess, "./save_model.ckpt")
    #
    # sess.run(prediction, feed_dict={xs:img.eval(),keep_prob:0.5})

    all_vars = tf.trainable_variables()
    print all_vars
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(all_vars)
    saver.restore(sess, save_path)

    while camera.isOpened():
        ret, frame = camera.read()
        frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
        frame = cv2.flip(frame, 1)                      # flip the frame horizontally
        # cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
        #              (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (0, 255, 0), 2)

        #  Main operation
        if isBgCaptured == 1:  # this part wont run until background captured
            #img = removeBG(frame)
            img = frame[0:int(cap_region_y_end * frame.shape[0]),
                        int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI

            cv2.imshow('mask', img)

            # convert the image into binary image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            blur_hsv = cv2.inRange(hsv, (0, 48, 80), (20, 255, 255))

            blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
            #blur_hsv= cv2.GaussianBlur(hsv, (blurValue, blurValue), 0)

            #cv2.imshow('blur', blur_hsv)
            ret, thresh = cv2.threshold(blur_hsv, threshold, 255, cv2.THRESH_BINARY)

            #--------------------------#

            cv2.imshow('ori', thresh)

            temp = cv2.resize(thresh, (32, 32), interpolation=cv2.INTER_AREA)
            img_t = tf.reshape(temp, [1, 32, 32])

            print img_t
            pred = sess.run(prediction, feed_dict={xs: img_t.eval(), keep_prob: 0.5})
            print pred[0]

            # get the coutours
            thresh1 = copy.deepcopy(thresh)
            _,contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            length = len(contours)
            maxArea = -1
            if length > 0:
                for i in range(length):  # find the biggest contour (according to area)
                    temp = contours[i]
                    area = cv2.contourArea(temp)
                    if area > maxArea:
                        maxArea = area
                        ci = i
                    else:
                        area=cv2.bitwise_not(area)
                res = contours[ci]
                hull = cv2.convexHull(res)
                drawing = np.zeros(img.shape, np.uint8)

                realHandLen = cv2.arcLength(res, True)
                handContour = cv2.approxPolyDP(res, 0.001 * realHandLen, True)
                minX, minY, handWidth, handHeight = cv2.boundingRect(handContour)

                cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
                cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

                cv2.rectangle(img, (minX, minY), (minX+handWidth, minY+handHeight), (255, 0, 0))

            cv2.imshow('rec',frame)

        # Keyboard OP
        k = cv2.waitKey(10)
        if k == 27:  # press ESC to exit
            break
        # elif k == ord('t'):
        #     counter+=1
        #     print 'picture captured!'
        #     print counter
        #     name='/users/cyk/Desktop/432/handGestureLib/point/point'+str(counter)+'.png'
        #     cv2.imwrite(name,thresh)
        #
        #     print 'finished!'
