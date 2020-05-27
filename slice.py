import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

x = tf.constant([[3, 2],
                 [4, 5],
                 [6, 7]])
print("x.shape", x.shape)

## [0, 1] for start position. slice 2 row and 1 column from the position
## [0,1]が開始位置。そこから2row, 1columを切り取る。
sliced = tf.slice(x, [0, 1], [2, 1])
print("sliced.shape", sliced.shape)


## tf.squeeze removes dimensions of size 1 from the shape of vector
## tf.squeezeは、ベクトルの形状からサイズ1の次元を削除します
## dimension (2,4,1) --> (2,4)
t = tf.constant([
    [[1],
    [2],
    [3],
    [4]],
    [[5],
    [6],
    [7],
    [8]],
])
squeezed = tf.squeeze(t)

## expand_dims
expanded = tf.expand_dims(x, axis=1)  # axis とは、何番目のdimensionにかということ。

with tf.compat.v1.Session() as sess:
    print("sliced:\n", sliced.eval())

    print("t.shape", t.shape)
    print("before squeeze:\n", t.eval())

    print("squeeze.shape", squeezed.shape)
    print("squeeze:\n", squeezed.eval())

    print("x.shape:\n", x.shape)
    print(x.eval())
    print("expanded.shape:\n:", expanded.shape)
    print(expanded.eval())


## ways for debug
""" 
tf.print
    tfdbg
    TensorBorad
    
    Change logging level
    tf.logging.set_verbosity(tf.logging.INFO)
"""



# def forward_pass(w, x):
#     return tf.matmul(w, x)
#
#
# def train_loop(x, niter=5):
#     with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):
#         w = tf.compat.v1.get_variable("weights",
#                                       shape=(1, 2),
#                                       initializer=tf.compat.v1.truncated_normal_initializer(),
#                                       trainable=True)
#     preds = []
#     for k in range(niter):
#         preds.append(forward_pass(w, x))
#         w = w + 0.1
#     return preds

# with tf.compat.v1.Session() as sess:
#     predictions = train_loop(tf.constant([[3.2, 5.1, 7.2],
#                                           [4.3, 6.2, 8.3]]))
#     tf.compat.v1.global_variables_initializer().run()
#
#     # predictions = [1, 2, 3]
#     for i in range(len(predictions)):
#         print("{}:{}".format(i, predictions[i].eval()))




