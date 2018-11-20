#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    graph = tf.get_default_graph()
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    input_layer = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer_3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer_4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer_7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    return input_layer, keep_prob, layer_3, layer_4, layer_7


tests.test_load_vgg(load_vgg, tf)


def create_1x1(input_layer, num_classes):
    return tf.layers.conv2d(input_layer, num_classes, kernel_size=(1, 1), strides=(1, 1), padding='same',
                            kernel_initializer=tf.truncated_normal_initializer(stddev=1e-2),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))


def create_upsampling_layer(input_layer, kernel_size, stride, n_classes):
    return tf.layers.conv2d_transpose(input_layer, n_classes, kernel_size, stride, padding='same',
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=1e-2),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # 1x1 connections
    # 1x1 layer before decoder
    conv1x1 = create_1x1(vgg_layer7_out, num_classes)
    # 1x1 to skip connections for layer 4
    conv1x1_layer_4 = create_1x1(vgg_layer4_out, num_classes)
    # 1x1 to skip connections for layer 3
    conv1x1_layer_3 = create_1x1(vgg_layer3_out, num_classes)

    output = create_upsampling_layer(conv1x1, 4, 2, num_classes)
    tf.add(output, conv1x1_layer_4)

    output = create_upsampling_layer(output, 4, 2, num_classes)
    output = tf.add(output, conv1x1_layer_3)

    output = create_upsampling_layer(output, 16, 8, num_classes)
    return output


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    regularization_constant = 0.01
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.reduce_mean(cross_entropy)
    loss = loss + regularization_constant * sum(reg_losses)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = optimizer.minimize(loss)
    return logits, optimizer, loss


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    keep_prob_value = .5
    learning_rate_value = .0001
    sess.run(tf.global_variables_initializer())
    for i in range(epochs):
        print('Epoch:', i)
        for X, y in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: X, correct_label: y, keep_prob: keep_prob_value,
                                          learning_rate: learning_rate_value})

            print('Loss', loss)


tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    n_epochs = 20
    batch_size = 8
    image_shape = (160, 576)
    data_dir = '/data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Load encoder layers needed to build decoder
        enc_input_layer, enc_keep_prob, enc_vgg_layer_3, enc_layer_4, enc_layer_7 = load_vgg(sess, vgg_path)

        # build decoder layers
        nn_output = layers(enc_vgg_layer_3, enc_layer_4, enc_layer_7, num_classes)

        # get optimizer, loss and logits tensors to perform training.
        correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        logits, optimizer, loss = optimize(nn_output, correct_label, learning_rate, num_classes)

        # train nn
        train_nn(sess, n_epochs, batch_size, get_batches_fn, optimizer, loss, enc_input_layer, correct_label,
                 enc_keep_prob, learning_rate)
        # save images
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, enc_keep_prob, enc_input_layer)


if __name__ == '__main__':
    run()
