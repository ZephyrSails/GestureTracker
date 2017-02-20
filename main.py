from handFinder import *
from cnn import *
from controller import *
import cv2
import time

def main():
    camera = cv2.VideoCapture(0)
    camera.set(10, 200)

    cnnPredictor = CNNPredictor()
    controller = Controller()

    lastTime = time.time()
    while camera.isOpened():

        print time.time() - lastTime
        lastTime = time.time()

        _, frame = camera.read()
        # cv2.imshow('frame', frame)

        contours, img, thresh = getContours(frame)

        if contours:
            img = getRectangle(contours, img, frame)

            temp = cv2.resize(thresh, (32, 32), interpolation=cv2.INTER_AREA)
            img_t = tf.reshape(temp, [1, 32, 32])
            #
            pred = cnnPredictor.predict(img_t)

            # cv2.imshow('img_t', img)

            controller.act(pred, [0, 0])

            # cv2.imshow('rec', frame)
        # pred = sess.run(prediction, feed_dict={xs: img_t.eval(), keep_prob: 0.5})
            print pred[0]

        k = cv2.waitKey(1)
        if k == 27:  # press ESC to exit
            break

if __name__ == '__main__':
    main()
