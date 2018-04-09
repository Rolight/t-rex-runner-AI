import tensorflow as tf


def build_mlp(input_placeholder,
              output_size,
              scope,
              n_layers=2,
              size=500,
              activation=tf.tanh,
              output_activation=None,
              kernel_regularizer=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(
                out, size, activation=activation, kernel_regularizer=kernel_regularizer)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out
