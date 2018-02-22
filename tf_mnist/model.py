import tensorflow as tf

ROOT_SCOPE = 'MNIST'


def mnist_model(inputs, training):
    """
    Create a (very simple) TF model to classify MNIST digits.

    :param inputs: a Tensor with the image inputs (N, 32, 32, 1)
    :param training: a bool or Tensor(tf.bool) indicating we're training or testing the model
    :return: the logits (pre-softmax) output Tensor
    """

    # We'll not use any contrib stuff, including slim.

    with tf.variable_scope(ROOT_SCOPE):
        net = tf.layers.conv2d(inputs, 32, 5, padding='same', activation=tf.nn.relu, name='conv1')
        net = tf.layers.max_pooling2d(net, 2, 2, name='pool1')
        net = tf.layers.conv2d(net, 64, 5, padding='same', activation=tf.nn.relu, name='conv2')
        net = tf.layers.max_pooling2d(net, 2, 2, name='pool2')
        net = tf.layers.flatten(net, name='flatten')
        net = tf.layers.dense(net, 1024, activation=tf.nn.relu, name='fc3')
        net = tf.layers.dropout(net, training)
        net = tf.layers.dense(net, 10, name='logits')

    return net
