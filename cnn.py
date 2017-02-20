import tensorflow as tf
import cv2


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


class CNNPredictor(object):
    def __init__(self):
        self.xs = tf.placeholder(tf.float32, shape=([1,32, 32]))
        self.x_image = tf.reshape(self.xs, [-1, 32, 32, 1])
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
        self.keep_prob = tf.placeholder(tf.float32)
        self.prediction = compute_net(self.x_image, weights, biases, self.keep_prob)


        self.sess = tf.Session()
        all_vars = tf.trainable_variables()
        # print all_vars
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(all_vars)
        saver.restore(self.sess, './save_model.ckpt')


    def predict(self, img_t):
        # print img_t
        # print img_t.eval()
        return self.sess.run(self.prediction, feed_dict={self.xs: img_t.eval(session=self.sess), self.keep_prob: 0.5})
