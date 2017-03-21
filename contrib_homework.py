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

x = tf.placeholder(tf.float32, shape=[None, 28*28], name='x') # groundtruth
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

def conv2d(x, W, shape):
  reshaped = tf.reshape(x, [-1,shape[0], shape[1],W.get_shape().as_list()[2]])
  conv = tf.nn.conv2d(reshaped, W, strides=[1, 1, 1, 1], padding='SAME')
  flattened = tf.reshape(conv, [-1, shape[0]*shape[1]]);
  return tf.nn.relu(flattened);

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def upsample_2d(x, size):
    a=x.get_shape().as_list()
    height = a[1]*size[0]
    width = a[2]*size[1]
    output_size = [height, width]
    return tf.image.resize_images(x, output_size,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

def dense_batch_relu(x, out, phase, scope):
    with tf.variable_scope(scope):
        print(x.get_shape())
        h1 = tf.contrib.layers.fully_connected(x, out,
                                               activation_fn=None,
                                               scope='dense', weights_initializer=tf.contrib.layers.initializers.xavier_initializer())
        h2 = tf.contrib.layers.batch_norm(h1,
                                          center=True, scale=True,
                                          is_training=phase,
                                          scope='bn')
        return tf.nn.relu(h2, 'relu')


tf.reset_default_graph()
#x_lr_vec = tf.reshape(x_lr, [-1,LOW_RES_SZ*LOW_RES_SZ])
x = tf.placeholder(tf.float32, shape=[None, 28*28], name='x')
x_image = tf.reshape(x, [-1,28,28,1])
x_lr = tf.image.resize_images(x_image, [LOW_RES_SZ,LOW_RES_SZ]) # low-resolution version of x
x_lr_vec = tf.reshape(x_lr, [-1,LOW_RES_SZ*LOW_RES_SZ])
phase = tf.placeholder(tf.bool, name='phase')



h1 = dense_batch_relu(x_lr_vec, 28*28, phase, 'layer1')
W_conv1 = weight_variable([3, 3, 1, 32])
W_conv3 = weight_variable([3, 3, 32, 1])

h_conv1 = conv2d(h1, W_conv1, [28,28]);
x_sr = conv2d(h_conv1, W_conv3,[28,28]);


'''
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

with tf.name_scope('loss'):
    reconstruct_error = tf.reduce_mean((x_sr-x)**2)

    loss = reconstruct_error;

#########################################################
def train():
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # Ensures that we execute the update_ops before performing the train_step
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    history = []
    iterep = 200
    epochs = 100
    for i in range(iterep * epochs):
        x_train = mnist.train.next_batch(64)
        sess.run(train_step,
                 feed_dict={'x:0':x_train[0],
                            'phase:0': 1})
        if (i + 1) %  iterep == 0:
            epoch = (i + 1)/iterep
            t = sess.run([loss],
                         feed_dict={x: mnist.test.images,
                                    'phase:0': 0})
            history += [[epoch] + t]
            print history[-1]
    return history, sess


history, sess = train()
#########################################################
# plot an example result
#########################################################
image_index = 21 # image index
print("showing %d-th validation image and the super-resolution output for it" % image_index)
for i in range(12):
    tmp_in = mnist.validation.images[image_index+i:image_index+1+i,:]
    tmp_lr = x_lr.eval(feed_dict={x:tmp_in, 'phase:0': 0},session=sess)
    tmp_out = x_sr.eval(feed_dict={x:tmp_in, 'phase:0': 0}, session=sess)
    plt.subplot(6,6,i*3+1)
    plt.imshow(tmp_in.reshape([28,28]), cmap='gray')
    plt.subplot(6,6,i*3+2)
    plt.imshow(tmp_lr.reshape([7,7]), cmap='gray')
    plt.subplot(6,6,i*3+3)
    plt.imshow(tmp_out.reshape([28,28]), cmap='gray')



#########################################################
# once you finalize the model, evaluate on the test set
# (only once, at the end, using the best model according the validation set error)
#########################################################
print("test error %g" % reconstruct_error.eval(feed_dict={
    x: mnist.test.images, 'phase:0':0}, session=sess))
plt.show()
