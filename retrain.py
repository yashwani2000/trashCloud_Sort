import sys
import numpy as np
import imageio
from PIL import Image
import onnx
import onnx_tf
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras

path = (sys.argv[1])
modelPath = (sys.argv[2])
net = onnx.load(modelPath)

#TODO : convert onnx to a tensorflow.keras model

#TODO : import dataset as a keras.Dataset

model.fit(data, labels, epochs=10, batch_size=32,
          validation_data=(val_data, val_labels))
