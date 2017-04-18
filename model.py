import tensorflow as tf
from ops import acts, layers

class MLP(object):
    def __init__(self):
        self.activation_fn = acts.lRelu
        self.leaky_val = 0.0
        self.hidden_nodes = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]

    def linear(self, input_, output_size, bias=0.0, name="linear"):
        input_size = input_.get_shape().as_list()[1]
        input_ = tf.cast(input_, tf.float32)
        with tf.variable_scope(name):
            w = tf.get_variable('w', [input_size, output_size], tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable('b', [output_size], initializer=tf.constant_initializer(bias))
            logits = tf.nn.xw_plus_b(input_, w, biases, name="logits")
            return logits


    def linear_act(self, input_, output_size, with_logit=False, name="linear_act"):
        with tf.variable_scope(name):
            logits = self.linear(input_, output_size)
            bn = layers.batch_norm(logits)
            act = self.activation_fn(bn, self.leaky_val)

            if with_logit:
                return act, logits
            else:
                return act


    def linear_repeat(self, input_, hidden_nodes, with_logit=True, name="linear_repeat"):
        num_repeat = len(hidden_nodes)

        with tf.variable_scope(name):
            output, logit = 0, 0

            for i in range(num_repeat):
                name = "linear_%d" % i
                output, logit = self.linear_act(input_, hidden_nodes[i], with_logit=True, name=name)
                input_ = output

            if with_logit:
                return output, logit
            else:
                return output


    def inference(self, input_):
        logits = self.linear_repeat(input_, self.hidden_nodes, with_logit=False, name='mlps')
        logit = self.linear(logits, 1, name='regression')

        return logit
