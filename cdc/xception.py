import logging
import tensorflow as tf
import cxflow_tensorflow as cxtf
import tensorflow.contrib.keras as K
import tensorflow.contrib.keras.api.keras.layers as layers
from tensorflow.contrib.keras.api.keras.layers import Dense, BatchNormalization, Activation, Conv2D, \
    SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout


class XCeptionNet(cxtf.BaseModel):
    """
    XCeption network adapted from Keras sources.

    Rerefences:
    <https://arxiv.org/pdf/1610.02357.pdf>
    <https://github.com/fchollet/keras/blob/master/keras/applications/xception.py>
    """

    def _create_model(self,
                      middle_flow_repeats: int=8,
                      dropout: float=0.,
                      weight_decay: float=0.,
                      **kwargs) -> None:
        """
        Craete XCeption model.

        :param middle_flow_repeats: number of middle flow block repeats
        :param dropout: dropout rate of the extracted features
        :param weight_decay: weight decay regularization
        """

        images = tf.placeholder(tf.float32, shape=[None] + self._dataset.shape, name='images')
        labels = tf.placeholder(tf.int64, shape=[None], name='labels')

        regularizer = K.regularizers.l2(weight_decay)

        with tf.variable_scope('model'):
            net = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1',
                         kernel_regularizer=regularizer)(images)
            net = BatchNormalization(name='block1_conv1_bn')(net, training=self.is_training)
            net = Activation('relu', name='block1_conv1_act')(net)
            net = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2', kernel_regularizer=regularizer)(net)
            net = BatchNormalization(name='block1_conv2_bn')(net, training=self.is_training)
            net = Activation('relu', name='block1_conv2_act')(net)

            residual = Conv2D(128, (1, 1), strides=(2, 2),
                              padding='same', use_bias=False, kernel_regularizer=regularizer)(net)
            residual = BatchNormalization()(residual, training=self.is_training)

            net = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1',
                                  depthwise_regularizer=regularizer, pointwise_regularizer=regularizer)(net)
            net = BatchNormalization(name='block2_sepconv1_bn')(net, training=self.is_training)
            net = Activation('relu', name='block2_sepconv2_act')(net)
            net = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2',
                                  depthwise_regularizer=regularizer, pointwise_regularizer=regularizer)(net)
            net = BatchNormalization(name='block2_sepconv2_bn')(net, training=self.is_training)

            net = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(net)
            net = layers.add([net, residual])

            residual = Conv2D(256, (1, 1), strides=(2, 2),
                              padding='same', use_bias=False, kernel_regularizer=regularizer)(net)
            residual = BatchNormalization()(residual, training=self.is_training)

            net = Activation('relu', name='block3_sepconv1_act')(net)
            net = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1',
                                  depthwise_regularizer=regularizer, pointwise_regularizer=regularizer)(net)
            net = BatchNormalization(name='block3_sepconv1_bn')(net, training=self.is_training)
            net = Activation('relu', name='block3_sepconv2_act')(net)
            net = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2',
                                  depthwise_regularizer=regularizer, pointwise_regularizer=regularizer)(net)
            net = BatchNormalization(name='block3_sepconv2_bn')(net, training=self.is_training)

            net = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(net)
            net = layers.add([net, residual])

            residual = Conv2D(728, (1, 1), strides=(2, 2),
                              padding='same', use_bias=False, kernel_regularizer=regularizer)(net)
            residual = BatchNormalization()(residual, training=self.is_training)

            net = Activation('relu', name='block4_sepconv1_act')(net)
            net = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1',
                                  depthwise_regularizer=regularizer, pointwise_regularizer=regularizer)(net)
            net = BatchNormalization(name='block4_sepconv1_bn')(net, training=self.is_training)
            net = Activation('relu', name='block4_sepconv2_act')(net)
            net = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2',
                                  depthwise_regularizer=regularizer, pointwise_regularizer=regularizer)(net)
            net = BatchNormalization(name='block4_sepconv2_bn')(net, training=self.is_training)

            net = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(net)
            net = layers.add([net, residual])

            for i in range(middle_flow_repeats):
                residual = net
                prefix = 'block' + str(i + 5)

                net = Activation('relu', name=prefix + '_sepconv1_act')(net)
                net = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1',
                                      depthwise_regularizer=regularizer, pointwise_regularizer=regularizer)(net)
                net = BatchNormalization(name=prefix + '_sepconv1_bn')(net, training=self.is_training)
                net = Activation('relu', name=prefix + '_sepconv2_act')(net)
                net = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2',
                                      depthwise_regularizer=regularizer, pointwise_regularizer=regularizer)(net)
                net = BatchNormalization(name=prefix + '_sepconv2_bn')(net, training=self.is_training)
                net = Activation('relu', name=prefix + '_sepconv3_act')(net)
                net = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3',
                                      depthwise_regularizer=regularizer, pointwise_regularizer=regularizer)(net)
                net = BatchNormalization(name=prefix + '_sepconv3_bn')(net, training=self.is_training)

                net = layers.add([net, residual])

            residual = Conv2D(1024, (1, 1), strides=(2, 2),
                              padding='same', use_bias=False, kernel_regularizer=regularizer)(net)
            residual = BatchNormalization()(residual, training=self.is_training)

            net = Activation('relu', name='block13_sepconv1_act')(net)
            net = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1',
                                  depthwise_regularizer=regularizer, pointwise_regularizer=regularizer)(net)
            net = BatchNormalization(name='block13_sepconv1_bn')(net, training=self.is_training)
            net = Activation('relu', name='block13_sepconv2_act')(net)
            net = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2',
                                  depthwise_regularizer=regularizer, pointwise_regularizer=regularizer)(net)
            net = BatchNormalization(name='block13_sepconv2_bn')(net, training=self.is_training)

            net = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(net)
            net = layers.add([net, residual])

            net = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1',
                                  depthwise_regularizer=regularizer, pointwise_regularizer=regularizer)(net)
            net = BatchNormalization(name='block14_sepconv1_bn')(net, training=self.is_training)
            net = Activation('relu', name='block14_sepconv1_act')(net)

            net = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2',
                                  depthwise_regularizer=regularizer, pointwise_regularizer=regularizer)(net)
            net = BatchNormalization(name='block14_sepconv2_bn')(net, training=self.is_training)
            net = Activation('relu', name='block14_sepconv2_act')(net)
            logging.info('Output shape: %s', net.shape)

        with tf.variable_scope('classifier'):
            net = GlobalAveragePooling2D(name='avg_pool')(net)
            if dropout > 0:
                net = Dropout(dropout)(net, training=self.is_training)
            logits = Dense(self._dataset.num_classes, activation=None)(net)

        # outputs
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        tf.identity(loss, name='loss')
        tf.nn.softmax(logits, 1, name='predictions')
        tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32, name='accuracy'))

    def _initialize_variables(self, finetune: str=None, **kwargs) -> None:
        """
        Initialize trainable variables randomly or from the given checkpoint.

        :param finetune: path to the checkpoint to finetune the variables from
        """
        if finetune is None:
            super()._initialize_variables(**kwargs)  # default initialization
        else:
            self._saver = tf.train.Saver(max_to_keep=100000000)
            logging.info('Restoring variables from `%s`', finetune)
            self._saver.restore(self.session, finetune)
