import os
import tensorflow as tf
from util import masked_softmax


class PolicyNetwork(object):
    """ Policy Function approximator. """

    def __init__(self, input_size, output_size, learning_rate=0.001, summaries_dir=None, scope="policy_estimator"):
        with tf.variable_scope(scope):
            # Writes Tensorboard summaries to disk
            self.summary_writer = None
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

            self.state = tf.placeholder(dtype=tf.float64, shape=[1, input_size], name="state")
            self.action = tf.placeholder(dtype=tf.int32, name="action")
            self.target = tf.placeholder(dtype=tf.float64, name="target")
            self.mask = tf.placeholder(dtype=tf.float64, shape=[1, output_size], name="mask")

            # This is just table lookup estimator
            # self.fc_layer1 = tf.contrib.layers.fully_connected(
            #      inputs=self.state,
            #      num_outputs=len(env.state),
            #      activation_fn=tf.nn.relu)

            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=self.state,
                num_outputs=output_size,
                activation_fn=None)

            # self.action_probs = tf.squeeze(tf.nn.softmax(self.output_layer))
            self.action_probs = tf.squeeze(masked_softmax(self.output_layer, self.mask))
            self.picked_action_prob = tf.gather(self.action_probs, self.action)

            # Loss and train op
            self.loss = -tf.log(self.picked_action_prob) * self.target

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.train.get_global_step())

    def predict(self, state, mask, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_probs, {self.state: state.reshape(1, -1),
                                            self.mask: mask.reshape(1, -1)})

    def update(self, state, target, action, mask, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state.reshape(1, -1), self.target: target,
                     self.action: action, self.mask: mask.reshape(1, -1)}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

    def restore(self, sess, checkpoint_file):
        sess = sess or tf.get_default_session()
        self.saver = tf.train.Saver(tf.global_variables())
        self.saver.restore(sess=sess, save_path=checkpoint_file)


class ValueNetwork(object):
    """ Value Function approximator. """

    def __init__(self, input_size, output_size=1, learning_rate=0.01, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(dtype=tf.float64, shape=[1, input_size], name="state")
            self.target = tf.placeholder(dtype=tf.float64, name="target")

            # This is just table lookup estimator
            # self.fc_layer1 = tf.contrib.layers.fully_connected(
            #     inputs=self.state,
            #     num_outputs=input_size,
            #     activation_fn=tf.nn.relu)

            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=self.state,
                num_outputs=output_size,
                activation_fn=None)

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.train.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, {self.state: state.reshape(1, -1)})

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state.reshape(1, -1), self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class ObjectAwareRewardNetwork(object):
    """ Object-aware Reward Function approximator. """

    def __init__(self, input_size, output_size, action_num, learning_rate=0.01, scope="reward_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(shape=[1, input_size], dtype=tf.float64, name="state")
            self.action = tf.placeholder(shape=[], dtype=tf.int32, name="question_idx")
            self.object = tf.placeholder(shape=[], dtype=tf.int32, name="person_idx")
            self.target = tf.placeholder(dtype=tf.float64, name="target")

            object_vec = tf.one_hot(self.object, input_size, dtype=tf.float64)
            action_vec = tf.one_hot(self.action, action_num, dtype=tf.float64)
            concat_vec = tf.concat([object_vec, action_vec], 0)

            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.concat([self.state, tf.expand_dims(concat_vec, 0)], 1),
                num_outputs=output_size,
                activation_fn=tf.nn.sigmoid)

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.train.get_global_step())

    def predict(self, state, action, object, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, {self.state: state.reshape(1, -1), self.action: action, self.object: object})

    def update(self, state, action, object, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state.reshape(1, -1), self.action: action, self.object: object, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)

    def restore(self, sess, checkpoint_file):
        sess = sess or tf.get_default_session()
        self.saver = tf.train.Saver(tf.global_variables())
        self.saver.restore(sess=sess, save_path=checkpoint_file)


class RewardNetwork(object):
    """ Reward Function approximator. """

    def __init__(self, input_size, output_size, action_num, learning_rate=0.01, scope="reward_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(shape=[1, input_size], dtype=tf.float64, name="state")
            self.action = tf.placeholder(shape=[], dtype=tf.int32, name="question_idx")
            self.target = tf.placeholder(dtype=tf.float64, name="target")

            action_vec = tf.one_hot(self.action, action_num, dtype=tf.float64)
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.concat([self.state, tf.expand_dims(action_vec, 0)], 1),
                num_outputs=output_size,
                activation_fn=tf.nn.sigmoid)

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.train.get_global_step())

    def predict(self, state, action, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, {self.state: state.reshape(1, -1), self.action: action})

    def update(self, state, action, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state.reshape(1, -1), self.action: action, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)

    def restore(self, sess, checkpoint_file):
        sess = sess or tf.get_default_session()
        self.saver = tf.train.Saver(tf.global_variables())
        self.saver.restore(sess=sess, save_path=checkpoint_file)

