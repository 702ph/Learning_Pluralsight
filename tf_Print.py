import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)



def some_method(a, b):
    b = tf.cast(b, tf.float32)
    s = (a/b)

    # print_ab = tf.compat.v1.Print(s, [a, b])
    # s = tf.where(tf.compat.v1.is_nan(s), print_ab, s)

    return tf.sqrt(tf.matmul(s, tf.transpose(s)))


fake_a = tf.constant([[5.0, 3.0, 7.1], [2.3, 4.1, 4.8]])
print(fake_a.shape)  # (2,3)

fake_b = tf.constant([[2, 0, 5], [2, 8, 7]])
print(fake_b.shape)  # (2,3)

with tf.compat.v1.Session() as sess:
    print(sess.run(some_method(fake_a, fake_b)))


