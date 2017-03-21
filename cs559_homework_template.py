# CS 559 homework template
# Instructor: Gokberk Cinbis
#
# Adapted from
# https://www.tensorflow.org/get_started/mnist/pros
# License: Apache 2.0 License
#
# Homework author: <Your name>

#########################################################
# preparations (you may not need to make changes here)
#########################################################
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# prepare data and TensorFlow
mnist = input_data.read_data_sets('data/MNIST_data', one_hot=True)
sess = tf.InteractiveSession()
LOW_RES_SZ = 7 # low-resolution input image size (constant)
beta = 0.001


# define input and down-sampler layers
x = tf.placeholder(tf.float32, shape=[None, 28*28]) # groundtruth
x_image = tf.reshape(x, [-1,28,28,1])
x_lr = tf.image.resize_images(x_image, [LOW_RES_SZ,LOW_RES_SZ]) # low-resolution version of x
keep_prob = tf.placeholder(tf.float32) # unused unless you have dropout regularization

#########################################################
# Define your super-resolution network here. The network
# should take 7x7 images as input (x_lr), and predict 28x28 super-resolution images (x_sr),
# (without using the 28x28 groundtruth image "x", or its class label).
# Below is a simple example that consists of a single linear layer.
#########################################################



def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def upsample_2d(x, size):
    a=x.get_shape().as_list()
    height = a[1]*size[0]
    width = a[2]*size[1]
    output_size = [height, width]
    return tf.image.resize_images(x, output_size,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    



x_lr_vec = tf.reshape(x_lr, [-1,LOW_RES_SZ*LOW_RES_SZ])
W1 = tf.Variable(tf.random_normal([LOW_RES_SZ*LOW_RES_SZ,28*28], stddev=0.1))
b1 = tf.Variable(tf.zeros([28*28]))

fc1 = tf.nn.relu(tf.matmul(x_lr_vec, W1) + b1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(fc1, keep_prob)
fcr1 = tf.reshape(h_fc1_drop, [-1, 28, 28, 1])
W_conv1 = weight_variable([3, 3, 1, 16])
W_conv3 = weight_variable([3, 3, 16, 1])
b_conv1 = bias_variable([16])
b_conv3 = bias_variable([1])
h_conv1 = tf.nn.relu(conv2d(fcr1, W_conv1) + b_conv1)
h_conv3 = tf.nn.relu(conv2d(h_conv1, W_conv3) + b_conv3)
x_sr = tf.reshape(h_conv3,[-1, 28*28])


'''
W_conv1 = weight_variable([3, 3, 1, 32])
W_conv2 = weight_variable([3, 3, 32, 32])
W_conv3 = weight_variable([3, 3, 32, 1])
W_conv4 = weight_variable([3, 3, 64, 1])
b_conv1 = bias_variable([32])
b_conv2 = bias_variable([32])
b_conv3 = bias_variable([1])
b_conv4 = bias_variable([1])
x_image_low = tf.reshape(x_lr, [-1,7,7,1])
print(x_image_low.get_shape())
h_conv1 = tf.nn.relu(conv2d(x_image_low, W_conv1) + b_conv1)
h_upsample1 = upsample_2d(h_conv1, (2,2));
h_conv2 = tf.nn.relu(conv2d(h_upsample1, W_conv2) + b_conv2)
h_upsample2 = upsample_2d(h_conv2, (2,2));
h_conv3 = tf.nn.relu(conv2d(h_upsample2, W_conv3) + b_conv3)
print(h_conv2.get_shape())
x_sr = tf.reshape(h_conv3,[-1, 28*28])
'''

print(x_sr.get_shape(), x.get_shape())
#########################################################
# train on the train subset of MNIST
#########################################################

# (write your training code here. here, we are just using the initial random weights)

#########################################################
# evaluate your model on the train and validation sets.
# use these two results when designing your architecture
# and tuning your optimization method.
#########################################################
reconstruct_error = tf.reduce_mean((x_sr-x)**2)

regularization = beta*(tf.nn.l2_loss(W1) + tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv3))

loss = reconstruct_error + regularization;

#########################################################
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.global_variables_initializer())
for i in range(10000):
  batch = mnist.train.next_batch(128)
  if i%100 == 0:
    train_accuracy = reconstruct_error.eval(feed_dict={
        x :batch[0], keep_prob: 1.0})
    print("step %d, reconstruction error %g"%(i, train_accuracy))
  train_step.run(feed_dict={ x:batch[0], keep_prob: 0.5})
#print("train error %g" % reconstruct_error.eval(feed_dict={
#    x: mnist.train.images, keep_prob: 1.0}))
print("val error %g" % reconstruct_error.eval(feed_dict={
    x: mnist.validation.images, keep_prob: 1.0}))

#########################################################
# plot an example result
#########################################################
image_index = 21 # image index
print("showing %d-th validation image and the super-resolution output for it" % image_index)
for i in range(12):
    tmp_in = mnist.validation.images[image_index+i:image_index+1+i,:]
    tmp_lr = x_lr.eval(feed_dict={x:tmp_in})
    tmp_out = x_sr.eval(feed_dict={x:tmp_in, keep_prob: 1.0})
    plt.subplot(6,6,i*3+1)
    plt.imshow(tmp_in.reshape([28,28]), cmap='gray')
    plt.subplot(6,6,i*3+2)
    plt.imshow(tmp_lr.reshape([7,7]), cmap='gray')
    plt.subplot(6,6,i*3+3)
    plt.imshow(tmp_out.reshape([28,28]), cmap='gray')

plt.show()

#########################################################
# once you finalize the model, evaluate on the test set
# (only once, at the end, using the best model according the validation set error)
#########################################################
print("test error %g" % reconstruct_error.eval(feed_dict={
    x: mnist.test.images, keep_prob: 1.0}))
