import tensorflow as tf


def forward_pass(w, x):
    return tf.matmul(w, x)


def train_loop(x, niter=5):
    with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):
        w = tf.compat.v1.get_variable("weights",
                                      shape=(1, 2),
                                      initializer=tf.compat.v1.truncated_normal_initializer(),
                                      trainable=True)
    preds = []
    for k in range(niter):
        preds.append(forward_pass(w, x))
        w = w + 0.1
    return preds


with tf.compat.v1.Session() as sess:
    predictions = train_loop(tf.constant([[3.2, 5.1, 7.2],
                                          [4.3, 6.2, 8.3]]))
    tf.compat.v1.global_variables_initializer().run()

    # predictions = [1, 2, 3]
    for i in range(len(predictions)):
        print("{}:{}".format(i, predictions[i].eval()))
