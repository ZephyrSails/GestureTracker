import os
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from tensorflow.core.protobuf import saver_pb2

# cwd = os.getcwd()

csvTrainRawFileName = 'train-labels.csv'
tfRecordTrainFileName = "train.tfrecords"

csvTestRawFileName = 'test-labels.csv'
tfRecordTestFileName = "test.tfrecords"

save_path="./save_model.ckpt"
batch_size=1

def compute_accuracy(v_xs, v_ys):
    global prediction

    y_pre = sess.run(prediction, feed_dict={xs:v_xs, keep_prob:0.5})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys, keep_prob:0.5 })
    return result

#input shape and return weight and bias
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

#x is the input(image or something), stride is the step width
def conv2d(x, W):
    #stride[1, x_movement, y_movement, 1]
    #Must have strides[4] = 1
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding='SAME')

#use pooling to handle big stride problem
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')








def create_record(file, outputName):
    writer = tf.python_io.TFRecordWriter(outputName)
    f = open(file, "r")
    for line in f:
        img_path, label = line.split(",")
        # print img_path, label
        img = Image.open(img_path)

        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        # print np.shape(img)
        #img = img.resize((32, 32))
        #img_raw = img.tobytes()
        img_raw = img.tobytes()

        print np.shape(img)

        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())
    writer.close()

def read_and_decode(filename):
    # print filename
    filename_queue = tf.train.string_input_producer([filename])
    #print filename_queue
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    #print serialized_example
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    # print img
    img = tf.reshape(img, [32, 32])
    #img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)


    return img, label

def compute_net(x, weights,biases,keep_prob):
    h_conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, weights['W_conv2']) + biases['b_conv2'])
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])  # 1 x 3136
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights['W_fc1']) + biases['b_fc1'])
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, weights['W_fc2']) + biases['b_fc2'])  # 1 x 1024
    # print prediction
    return prediction
    

if __name__ == '__main__':
    # create_record(csvTrainRawFileName, tfRecordTrainFileName)
    # create_record(csvTestRawFileName, tfRecordTestFileName)

    if True:
        xs = tf.placeholder(tf.float32, shape=([None,32, 32]))
        ys = tf.placeholder(tf.float32, shape=([None,3,]))
        keep_prob = tf.placeholder(tf.float32)
        x_image = tf.reshape(xs, [-1,32,32,1])



        weights = {
            'W_conv1': weight_variable([5, 5, 1, 32]),
            'W_conv2': weight_variable([5, 5, 32, 64]),
            'W_fc1': weight_variable([8 * 8 * 64, 1024]),
            'W_fc2': weight_variable([1024, 3])

        }
        biases = {
            'b_conv1': bias_variable([32]),
            'b_conv2': bias_variable([64]),
            'b_fc1': bias_variable([1024]),
            'b_fc2': bias_variable([3])
        }

        # convolutions = {
        #     'h_conv1': tf.Variable(tf.zeros([None, 32, 32, 32]), dtype=tf.float32),
        #     'h_conv2': tf.Variable(tf.zeros([None, 16, 16, 64]), dtype=tf.float32)
        # }
        #
        # poolings = {
        #     'h_pool1': tf.Variable(tf.zeros[None, 16, 16, 32], dtype=tf.float32),
        #     'h_pool2': tf.Variable(tf.zeros([None, 8, 8, 64]), dtype=tf.float32)
        # }

        prediction = compute_net(x_image, weights,biases,keep_prob)

        # the error between prediction and real data
        # cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
        #                                                reduction_indices=[1]))  # loss
        cross_entropy = -tf.reduce_sum(ys * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


        img_train, label_train = read_and_decode(tfRecordTrainFileName)
        img_train_batch, label_train_batch = tf.train.shuffle_batch([img_train, label_train],
                                                         batch_size=200, capacity=2000,
                                                         min_after_dequeue=1000)


        img_test, label_test = read_and_decode(tfRecordTestFileName)
        img_test_batch, label_test_batch = tf.train.shuffle_batch([img_train, label_train],
                                                                     batch_size=600, capacity=2000,
                                                                     min_after_dequeue=1000)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()


        with tf.Session() as sess:
            sess.run(init)
            threads = tf.train.start_queue_runners(sess=sess)

            for i in range(1000):
                #print img_train_batch
                img_temp, label_temp = sess.run([img_train_batch, label_train_batch])
                label_temp=sess.run(tf.one_hot(label_temp,3,1,0))
                sess.run(train_step, feed_dict={xs: img_temp, ys: label_temp, keep_prob: 0.5})

                if i % 20 == 0:
                    img_temp_t, label_temp_t = sess.run([img_test_batch, label_test_batch])
                    label_temp_t = sess.run(tf.one_hot(label_temp_t, 3, 1, 0))
                    print compute_accuracy(img_temp_t, label_temp_t)

                    saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V1)
                    saver.save(sess, 'save_model.ckpt')
                    # saved = saver.save(sess, save_path)
                    print("Model saved in file: %s" % save_path)
