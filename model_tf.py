import numpy as np
import tensorflow as tf


def compute_pure_ratio(ind1, ind2, indices, noise_or_not):
    num_remember = len(ind1)
    pure_ratio_1 = np.sum(noise_or_not[indices[ind1]]) / float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[indices[ind2]]) / float(num_remember)
    return pure_ratio_1, pure_ratio_2


def conv_layer(inputs, filters, kernel_size, strides, padding, training, reuse, name="conv_layer"):
    with tf.variable_scope(name, reuse=reuse):
        conv = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
        conv = tf.layers.batch_normalization(conv, momentum=0.9, epsilon=1e-5, training=training)
        conv = tf.nn.leaky_relu(conv, alpha=0.01)
        return conv


def cnn_model(images, n_outputs, drop_rate, training, top_bn=False, reuse=None, name="cnn_model"):
    # same model as used in the PyTorch version, can be any model theoretically。
    with tf.variable_scope(name, reuse=reuse):
        conv1 = conv_layer(images, filters=128, kernel_size=3, strides=1, padding="same", training=training,
                           reuse=None, name="conv_layer_1")
        conv2 = conv_layer(conv1, filters=128, kernel_size=3, strides=1, padding="same", training=training,
                           reuse=None, name="conv_layer_2")
        conv3 = conv_layer(conv2, filters=128, kernel_size=3, strides=1, padding="same", training=training,
                           reuse=None, name="conv_layer_3")
        pool3 = tf.layers.max_pooling2d(conv3, pool_size=2, strides=2)
        drop3 = tf.layers.dropout(pool3, rate=drop_rate, training=training,
                                  noise_shape=[tf.shape(pool3)[0], tf.shape(pool3)[1], tf.shape(pool3)[2], 1])

        conv4 = conv_layer(drop3, filters=256, kernel_size=3, strides=1, padding="same", training=training,
                           reuse=None, name="conv_layer_4")
        conv5 = conv_layer(conv4, filters=256, kernel_size=3, strides=1, padding="same", training=training,
                           reuse=None, name="conv_layer_5")
        conv6 = conv_layer(conv5, filters=256, kernel_size=3, strides=1, padding="same", training=training,
                           reuse=None, name="conv_layer_6")
        pool6 = tf.layers.max_pooling2d(conv6, pool_size=2, strides=2)
        drop6 = tf.layers.dropout(pool6, rate=drop_rate, training=training,
                                  noise_shape=[tf.shape(pool6)[0], tf.shape(pool6)[1], tf.shape(pool6)[2], 1])

        conv7 = conv_layer(drop6, filters=512, kernel_size=3, strides=1, padding="valid", training=training,
                           reuse=None, name="conv_layer_7")
        conv8 = conv_layer(conv7, filters=256, kernel_size=3, strides=1, padding="valid", training=training,
                           reuse=None, name="conv_layer_8")
        conv9 = conv_layer(conv8, filters=128, kernel_size=3, strides=1, padding="valid", training=training,
                           reuse=None, name="conv_layer_9")
        conv9_shape = conv9.get_shape().as_list()
        pool9 = tf.layers.average_pooling2d(conv9, pool_size=conv9_shape[1], strides=conv9_shape[1])

        h = tf.layers.flatten(pool9)
        h = tf.layers.dense(h, units=n_outputs, use_bias=True)
        if top_bn:
            h = tf.layers.batch_normalization(h, momentum=0.9, epsilon=1e-5, training=training)
        logits = h
        predicts = tf.argmax(tf.nn.softmax(h, axis=-1), axis=-1)
        return logits, predicts


def coteach_loss(logits1, logits2, labels, forget_rate):
    # compute loss
    raw_loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits1, labels=labels)
    raw_loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2, labels=labels)

    # sort and get low loss indices
    ind1_sorted = tf.argsort(raw_loss1, axis=-1, direction="ASCENDING", stable=True)
    ind2_sorted = tf.argsort(raw_loss2, axis=-1, direction="ASCENDING", stable=True)
    num_remember = tf.cast((1.0 - forget_rate) * ind1_sorted.shape[0].value, dtype=tf.int32)
    ind1_update = ind1_sorted[:num_remember]
    ind2_update = ind2_sorted[:num_remember]

    # update logits and compute loss again
    logits1_update = tf.gather(logits1, ind2_update, axis=0)
    labels1_update = tf.gather(labels, ind2_update, axis=0)
    loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits1_update, labels=labels1_update)
    loss1 = tf.reduce_sum(loss1) / tf.cast(num_remember, dtype=tf.float32)

    logits2_update = tf.gather(logits2, ind1_update, axis=0)
    labels2_update = tf.gather(labels, ind1_update, axis=0)
    loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2_update, labels=labels2_update)
    loss2 = tf.reduce_sum(loss2) / tf.cast(num_remember, dtype=tf.float32)

    return loss1, loss2, ind1_update, ind2_update


class CoTeachingModel:
    def __init__(self, input_shape, n_outputs, batch_size=128, drop_rate=0.25, top_bn=False):
        self.input_shape = input_shape
        self.n_outputs = n_outputs
        self.batch_size = batch_size
        self.drop_rate = drop_rate
        self.top_bn = top_bn
        self._add_placeholder()
        self._build_network()

    def _add_placeholder(self):
        self.images = tf.placeholder(name="images", shape=[self.batch_size] + self.input_shape, dtype=tf.float32)
        self.labels = tf.placeholder(name="labels", shape=[self.batch_size], dtype=tf.int32)
        self.training = tf.placeholder(name="training", shape=[], dtype=tf.bool)
        self.lr = tf.placeholder(name="learning_rate", shape=[], dtype=tf.float32)
        self.forget_rate = tf.placeholder(name="forget_rate", shape=[], dtype=tf.float32)

    def _build_network(self):
        logits1, self.predicts1 = cnn_model(self.images, self.n_outputs, self.drop_rate, self.training, self.top_bn,
                                            reuse=None, name="cnn_model_1")
        logits2, self.predicts2 = cnn_model(self.images, self.n_outputs, self.drop_rate, self.training, self.top_bn,
                                            reuse=None, name="cnn_model_2")

        self.acc1 = tf.reduce_mean(tf.cast(tf.equal(self.predicts1, tf.cast(self.labels, dtype=tf.int64)),
                                           dtype=tf.float32))
        self.acc2 = tf.reduce_mean(tf.cast(tf.equal(self.predicts2, tf.cast(self.labels, dtype=tf.int64)),
                                           dtype=tf.float32))

        # co-teaching loss
        self.loss1, self.loss2, self.ind1_update, self.ind2_update = coteach_loss(logits1, logits2, self.labels,
                                                                                  self.forget_rate)

        # trainable variables
        model1_vars = [x for x in tf.trainable_variables(scope="cnn_model_1")]
        model2_vars = [x for x in tf.trainable_variables(scope="cnn_model_2")]

        # update ops of batch normalization
        self.extra_update_ops1 = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="cnn_model_1")
        self.extra_update_ops2 = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="cnn_model_2")

        # create train operations
        self.train_op1 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss1, var_list=model1_vars)
        self.train_op2 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss2, var_list=model2_vars)


class BaseModel:
    def __init__(self, input_shape, n_outputs, batch_size=128, drop_rate=0.25, top_bn=False):
        self.input_shape = input_shape
        self.n_outputs = n_outputs
        self.batch_size = batch_size
        self.drop_rate = drop_rate
        self.top_bn = top_bn
        self._add_placeholder()
        self._build_network()

    def _add_placeholder(self):
        self.images = tf.placeholder(name="images", shape=[self.batch_size] + self.input_shape, dtype=tf.float32)
        self.labels = tf.placeholder(name="labels", shape=[self.batch_size], dtype=tf.int32)
        self.training = tf.placeholder(name="training", shape=[], dtype=tf.bool)
        self.lr = tf.placeholder(name="learning_rate", shape=[], dtype=tf.float32)

    def _build_network(self):
        logits, self.predicts = cnn_model(self.images, self.n_outputs, self.drop_rate, self.training, self.top_bn,
                                          reuse=None, name="cnn_model")
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predicts, tf.cast(self.labels, dtype=tf.int64)),
                                               dtype=tf.float32))
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)
        model_vars = [x for x in tf.trainable_variables(scope="cnn_model")]
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, var_list=model_vars)
