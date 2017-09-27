import logging
import tensorflow as tf
import tensorflow.contrib.slim as slim
import cxflow_tensorflow as cxtf


class CDCNaiveNet(cxtf.BaseModel):
    """
    Simple VGG-like net for <https://www.kaggle.com/c/cdiscount-image-classification-challenge>.

    Reference <https://arxiv.org/pdf/1409.1556.pdf>.
    Code <https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim>
    """

    def _create_model(self):
        # inputs
        images = tf.placeholder(tf.float32, shape=[None]+self._dataset.shape, name='images')
        labels = tf.placeholder(tf.int64, shape=[None], name='labels')

        # model
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            reuse=tf.get_variable_scope().reuse):
            with slim.arg_scope([slim.conv2d], padding='same'):
                net = slim.repeat(images, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                # net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                # net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                net = slim.max_pool2d(net, [2, 2], scope='pool5')
            net = slim.flatten(net)
            logging.info('Flatten shape `{}`'.format(net.shape))
            net = slim.fully_connected(net, 4096, scope='fc6')
            net = slim.dropout(net, 0.5, scope='dropout6', is_training=self.is_training)
            net = slim.fully_connected(net, 4096, scope='fc7')
            net = slim.dropout(net, 0.5, scope='dropout7', is_training=self.is_training)
            logits = slim.fully_connected(net, self._dataset.num_classes, activation_fn=None, scope='fc8')
        logging.info('Output shape `{}`'.format(logits.shape))

        # outputs
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        tf.identity(loss, name='loss')
        tf.nn.softmax(logits, 1, name='predictions')
        tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32, name='accuracy'))
