#!/usr/bin/env python
# coding: utf-8

# # MNIST Image Classification with TensorFlow on Cloud ML Engine
# 
# This notebook demonstrates how to implement different image models on MNIST using Estimator. 
# 
# Note the MODEL_TYPE; change it to try out different models

# In[3]:


import os
PROJECT = "qwiklabs-gcp-04-4860b2abb9b5" # REPLACE WITH YOUR PROJECT ID
BUCKET = "qwiklabs-gcp-04-4860b2abb9b5" # REPLACE WITH YOUR BUCKET NAME
REGION = "us-central1" # REPLACE WITH YOUR BUCKET REGION e.g. us-central1
MODEL_TYPE = "dnn"  # "linear", "dnn", "dnn_dropout", or "cnn"

# Do not change these
os.environ["PROJECT"] = PROJECT
os.environ["BUCKET"] = BUCKET
os.environ["REGION"] = REGION
os.environ["MODEL_TYPE"] = MODEL_TYPE
os.environ["TFVERSION"] = "1.13"  # Tensorflow version


# In[4]:


get_ipython().run_cell_magic('bash', '', 'gcloud config set project $PROJECT\ngcloud config set compute/region $REGION')


# ## Run as a Python module
# 
# In the previous notebook (mnist_linear.ipynb) we ran our code directly from the notebook.
# 
# Now since we want to run our code on Cloud ML Engine, we've packaged it as a python module.
# 
# The `model.py` and `task.py` containing the model code is in <a href="mnistmodel/trainer">mnistmodel/trainer</a>
# 
# **Complete the TODOs in `model.py` before proceeding!**
# 
# Once you've completed the TODOs, set MODEL_TYPE and run it locally for a few steps to test the code.

# In[3]:


get_ipython().run_cell_magic('bash', '', 'rm -rf mnistmodel.tar.gz mnist_trained\ngcloud ai-platform local train \\\n    --module-name=trainer.task \\\n    --package-path=${PWD}/mnistmodel/trainer \\\n    -- \\\n    --output_dir=${PWD}/mnist_trained \\\n    --train_steps=100 \\\n    --learning_rate=0.01 \\\n    --model=$MODEL_TYPE')


# **Now, let's do it on Cloud ML Engine so we can train on GPU:** `--scale-tier=BASIC_GPU`
# 
# Note the GPU speed up depends on the model type. You'll notice the more complex CNN model trains significantly faster on GPU, however the speed up on the simpler models is not as pronounced.

# In[4]:


get_ipython().run_cell_magic('bash', '', 'OUTDIR=gs://${BUCKET}/mnist/trained_${MODEL_TYPE}\nJOBNAME=mnist_${MODEL_TYPE}_$(date -u +%y%m%d_%H%M%S)\necho $OUTDIR $REGION $JOBNAME\ngsutil -m rm -rf $OUTDIR\ngcloud ai-platform jobs submit training $JOBNAME \\\n    --region=$REGION \\\n    --module-name=trainer.task \\\n    --package-path=${PWD}/mnistmodel/trainer \\\n    --job-dir=$OUTDIR \\\n    --staging-bucket=gs://$BUCKET \\\n    --scale-tier=BASIC_GPU \\\n    --runtime-version=$TFVERSION \\\n    -- \\\n    --output_dir=$OUTDIR \\\n    --train_steps=10000 --learning_rate=0.01 --train_batch_size=512 \\\n    --model=$MODEL_TYPE --batch_norm')


# ## Monitor training with TensorBoard
# 
# To activate TensorBoard within the JupyterLab UI navigate to "<b>File</b>" - "<b>New Launcher</b>".   Then double-click the 'Tensorboard' icon on the bottom row.
# 
# TensorBoard 1 will appear in the new tab.  Navigate through the three tabs to see the active TensorBoard.   The 'Graphs' and 'Projector' tabs offer very interesting information including the ability to replay the tests.
# 
# You may close the TensorBoard tab when you are finished exploring.

# ## Deploying and predicting with model
# 
# Deploy the model:

# In[1]:


get_ipython().run_cell_magic('bash', '', 'ls')


# In[9]:


get_ipython().run_cell_magic('bash', '', 'MODEL_NAME="mnist"\nMODEL_VERSION=${MODEL_TYPE}\nMODEL_LOCATION=$(gsutil ls gs://${BUCKET}/mnist/trained_${MODEL_TYPE}/export/exporter | tail -1)\necho "Deleting and deploying $MODEL_NAME $MODEL_VERSION from $MODEL_LOCATION ... this will take a few minutes"\n#何回も試す時は以前のを削除すること！\n#gcloud ai-platform versions delete ${MODEL_VERSION} --model ${MODEL_NAME}\n#gcloud ai-platform models delete ${MODEL_NAME}\ngcloud ai-platform models create ${MODEL_NAME} --regions $REGION\ngcloud ai-platform versions create ${MODEL_VERSION} --model ${MODEL_NAME} --origin ${MODEL_LOCATION} --runtime-version=$TFVERSION')


# To predict with the model, let's take one of the example images.

# In[14]:


import json, codecs
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

HEIGHT = 28
WIDTH = 28

mnist = input_data.read_data_sets("mnist/data", one_hot = True, reshape = False)
IMGNO = 7 #CHANGE THIS to get different images
jsondata = {"image": mnist.test.images[IMGNO].reshape(HEIGHT, WIDTH).tolist()}
json.dump(jsondata, codecs.open("test.json", "w", encoding = "utf-8"))
plt.imshow(mnist.test.images[IMGNO].reshape(HEIGHT, WIDTH));


# Send it to the prediction service

# In[15]:


get_ipython().run_cell_magic('bash', '', 'gcloud ai-platform predict \\\n    --model=mnist \\\n    --version=${MODEL_TYPE} \\\n    --json-instances=./test.json')


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
