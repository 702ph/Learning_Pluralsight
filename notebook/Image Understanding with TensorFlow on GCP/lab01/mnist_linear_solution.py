#!/usr/bin/env python
# coding: utf-8

# # MNIST Image Classification with TensorFlow
# 
# This notebook demonstrates how to implement a simple linear image models on MNIST using Estimator.
# <hr/>
# This <a href="mnist_models.ipynb">companion notebook</a> extends the basic harness of this notebook to a variety of models including DNN, CNN, dropout, pooling etc.

# In[1]:


import numpy as np
import shutil
import os
import tensorflow as tf
print(tf.__version__)


# ## Exploring the data
# 
# Let's download MNIST data and examine the shape. We will need these numbers ...

# In[2]:


from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist/data", one_hot = True, reshape = False)
print(mnist.train.images.shape)
print(mnist.train.labels.shape)


# In[3]:


HEIGHT = 28
WIDTH = 28
NCLASSES = 10


# In[4]:


import matplotlib.pyplot as plt
IMGNO = 12
plt.imshow(mnist.test.images[IMGNO].reshape(HEIGHT, WIDTH));


# ## Define the model.
# Let's start with a very simple linear classifier. All our models will have this basic interface -- they will take an image and return logits.

# In[5]:


# Using low-level tensorflow
def linear_model(img):
    X = tf.reshape(tensor = img, shape = [-1, HEIGHT * WIDTH]) #flatten
    W = tf.get_variable(name = "W", shape = [HEIGHT * WIDTH, NCLASSES], initializer = tf.truncated_normal_initializer(stddev = 0.1, seed = 1))
    b = tf.get_variable(name = "b", shape = [NCLASSES], initializer = tf.zeros_initializer)
    ylogits = tf.matmul(a = X, b = W) + b
    return ylogits, NCLASSES


# Note we can also build our linear classifer using the tf.layers API. Notice when using tf.layers we don't have to define or initialize our weights and biases. This happens automatically for us in the background.
# 
# When building more complex models such as DNNs and CNNs our code will be much more readable by using the tf.layers API

# In[6]:


# Using tf.layers API
def linear_model(img):
    X = tf.reshape(tensor = img, shape = [-1, HEIGHT * WIDTH]) #flatten
    ylogits = tf.layers.dense(inputs = X, units = NCLASSES, activation = None)
    return ylogits, NCLASSES


# ## Write Input Functions
# 
# As usual, we need to specify input functions for training, evaluation, and predicition.

# In[7]:


train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {"image": mnist.train.images},
    y = mnist.train.labels,
    batch_size = 100,
    num_epochs = None,
    shuffle = True,
    queue_capacity = 5000
)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {"image": mnist.test.images},
    y = mnist.test.labels,
    batch_size = 100,
    num_epochs = 1,
    shuffle = False,
    queue_capacity = 5000
)

def serving_input_fn():
    inputs = {"image": tf.placeholder(dtype = tf.float32, shape = [None, HEIGHT, WIDTH])}
    features = inputs # as-is
    return tf.estimator.export.ServingInputReceiver(features = features, receiver_tensors = inputs)


# ## Write Custom Estimator
# I could have simply used a canned LinearClassifier, but later on, I will want to use different models, and so let's write a custom estimator

# In[8]:


def image_classifier(features, labels, mode, params):
    ylogits, nclasses = linear_model(features["image"])
    probabilities = tf.nn.softmax(logits = ylogits)
    class_ids = tf.cast(x = tf.argmax(input = probabilities, axis = 1), dtype = tf.uint8)

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.reduce_mean(input_tensor = tf.nn.softmax_cross_entropy_with_logits_v2(logits = ylogits, labels = labels))
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.contrib.layers.optimize_loss(
                loss = loss, 
                global_step = tf.train.get_global_step(),
                learning_rate = params["learning_rate"], 
                optimizer = "Adam")
            eval_metric_ops = None
        else:
            train_op = None
            eval_metric_ops =  {"accuracy": tf.metrics.accuracy(labels = tf.argmax(input = labels, axis = 1), predictions = class_ids)}
    else:
        loss = None
        train_op = None
        eval_metric_ops = None
 
    return tf.estimator.EstimatorSpec(
        mode = mode,
        predictions = {"probabilities": probabilities, "class_ids": class_ids},
        loss = loss,
        train_op = train_op,
        eval_metric_ops = eval_metric_ops,
        export_outputs = {"predictions": tf.estimator.export.PredictOutput({"probabilities": probabilities, "class_ids": class_ids})}
    )


#  tf.estimator.train_and_evaluate does distributed training.

# In[9]:


def train_and_evaluate(output_dir, hparams):
    estimator = tf.estimator.Estimator(
        model_fn = image_classifier,
        model_dir = output_dir,
        params = hparams)

    train_spec = tf.estimator.TrainSpec(
        input_fn = train_input_fn,
        max_steps = hparams["train_steps"])

    exporter = tf.estimator.LatestExporter(name = "exporter", serving_input_receiver_fn = serving_input_fn)

    eval_spec = tf.estimator.EvalSpec(
        input_fn = eval_input_fn,
        steps = None,
        exporters = exporter)

    tf.estimator.train_and_evaluate(estimator = estimator, train_spec = train_spec, eval_spec = eval_spec)


# This is the main() function

# In[10]:


OUTDIR = "mnist/learned"
shutil.rmtree(path = OUTDIR, ignore_errors = True) # start fresh each time

hparams = {"train_steps": 1000, "learning_rate": 0.01}
train_and_evaluate(OUTDIR, hparams)


# I got:
# 
# `Saving dict for global step 1000: accuracy = 0.9158, global_step = 1000, loss = 0.29720208`
# 
# In other words, we achieved 91.6% accuracy with the simple linear model!

# <pre>
# # Copyright 2017 Google Inc. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #      http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# </pre>
