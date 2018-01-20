
import numpy as np
import tensorflow as tf
from constants import FLAGS
from time import time


class Predictor:
    def __init__(self, corpus, corpus_test=None):
        self.corpus = corpus
        self.corpus_test = corpus  # TODO change this to corpus_test
        self.setup_placeholders()
        self.setup_train_op()
        self.run_name = int(time())  # Just the epoch time
        print("Run name: {}".format(self.run_name))
        self.setup_saver()
        self.y_out = self.build_network(self.X, self.is_training)
        self.setup_loss(self.y_out)

    def setup_placeholders(self):
        self.X = tf.placeholder(tf.float32, shape=[None, FLAGS.image_x, FLAGS.image_y, FLAGS.image_c], name="X")
        self.y = tf.placeholder(tf.float32, shape=[None, FLAGS.image_x, FLATS.image_y, 1], name="y")
        self.is_training = tf.placeholder(tf.bool, shape=(), name="is_training")
        self.lr = tf.placeholder(tf.float32, shape=(), name="lr")
        self.Variable = tf.placeholder(int(0), trainable=False, name="global_step")

    def get_optimizer(self, lr):
        return tf.train.AdamOptimizer(lr)

    def setup_train_op(self, lr, step, loss):
        with tf.variable_scope('train_op'):
            optimizer = self.get_optimizer(lr)
            grads_and_vars = optimizer.compute_gradients(
                loss, tf.trainable_variables()
            )
            grads = [g for g, v in grads_and_vars]
            global_norm = tf.global_norm(grads)

            # Batch Norm in tensorflow requires this extra dependency
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.apply_gradients(
                    grads_and_vars, global_step=step
                )
            return train_op, global_norm

    def setup_saver(self):
        self.saver = tf.train.Saver()
        self.save_dir = os.path.join(
            FLAGS.models_dir, self.run_name
        )
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def setup_loss(self, y):
        # TODO: Finish this
        with tf.variable_scope('loss'):
            # TODO: Change this to unet loss
            l = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y, logits=self.raw_scores
            )
            loss = tf.reduce_mean(l)
            return loss

    def build_network(self, X, y, is_training):
        raise NotImplementedError()

    def restore(self, run_name):
        self.saver.restore(
                self.sess, os.path.join(FLAGS.models_dir, run_name)
            )
        print("Run {} restored.".format(run_name))
        print("Global Step: {}".format(self.global_step.eval()))

    def train(self):
        # train = self.corpus.get_list_images_and_masks()
        raise NotImplementedError()

    def predict(self):
        for example in self.corpus_test.get_examples():
            prediction = np.zeros_like(example.image)
            example.set_prediction(prediction)

    def write(self, filename):
        self.corpus_test.generate_submission(filename)


class UNet(Predictor):
    def __init__(self):
        super().__init__()

    def train(self):
        train = self.corpus.get_list_images_and_masks()
        do_something = 0

    def build_network(self, X, is_training):
        with tf.variable_scope('u-net'):
            print("X", X.shape)
            n = 64
            c1 = layers.conv2d(X, filters=n, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu)
            c1 = layers.conv2d(c1, filters=n, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu)
            p1 = layers.max_pooling2d(c1, pool_size=[2, 2], strides=[2, 2])
            print("p1", p1.shape)

            n *= 2
            c2 = layers.conv2d(p1, filters=n, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu)
            c2 = layers.conv2d(c2, filters=n, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu)
            p2 = layers.max_pooling2d(c2, pool_size=[2, 2], strides=[2, 2])
            print("p2", p2.shape)

            n *= 2
            c3 = layers.conv2d(p2, filters=n, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu)
            c3 = layers.conv2d(c3, filters=n, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu)
            p3 = layers.max_pooling2d(c3, pool_size=[2, 2], strides=[2, 2])
            print("p3", p3.shape)

            n *= 2
            c4 = layers.conv2d(p3, filters=n, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu)
            c4 = layers.conv2d(c4, filters=n, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu)
            p4 = layers.max_pooling2d(c4, pool_size=[2, 2], strides=[2, 2])
            print("p4", p4.shape)

            n *= 2
            c5 = layers.conv2d(p4, filters=n, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu)
            c5 = layers.conv2d(c5, filters=n, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu)
            print("c5", c5.shape)

            n /= 2
            u6 = layers.conv2d_transpose(c5, filters=n, kernel_size=[2, 2], padding='valid')
            u6 = tf.concat([u6, c4])
            print("u6", u6.shape)
            c6 = layers.conv2d(u6, filters=n, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu)
            c6 = layers.conv2d(c6, filters=n, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu)
            print("c6", c6.shape)

            n /= 2
            u7 = layers.conv2d_transpose(c6, filters=n, kernel_size=[2, 2], padding='valid')
            u7 = tf.concat([u7, c3])
            print("u7", u7.shape)
            c7 = layers.conv2d(u7, filters=n, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu)
            c7 = layers.conv2d(c7, filters=n, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu)
            print("c7", c7.shape)

            n /= 2
            u8 = layers.conv2d_transpose(c7, filters=n, kernel_size=[2, 2], padding='valid')
            u8 = tf.concat([u8, c2])
            print("u8", u8.shape)
            c8 = layers.conv2d(u8, filters=n, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu)
            c8 = layers.conv2d(c8, filters=n, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu)
            print("c8", c8.shape)

            n /= 2
            u9 = layers.conv2d_transpose(c8, filters=n, kernel_size=[2, 2], padding='valid')
            u9 = tf.concat([u9, c1])  # Axis = 3?
            print("u9", u9.shape)
            c9 = layers.conv2d(u9, filters=n, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu)
            c9 = layers.conv2d(c9, filters=n, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu)
            print("c9", c9.shape)

            output = layers.conv2d(c9, filters=1, kernel_size=[1, 1], activation=tf.sigmoid)
            print("output", output.shape)
            return output
