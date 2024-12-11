
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%   This file is part of https://github.com/verlab/GeoPatch_CVIU_2022
#
#   geopatch-descriptor is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   geopatch-descriptor is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with geopatch-descriptor.  If not, see <http://www.gnu.org/licenses/>.
#%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import cv2
import math

from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json

import os
###############################################################
#Configure Session for Memory Limit on TensorFlow using Keras
# session config
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))
###############################################################

pretrained_path = os.path.dirname(os.path.realpath(__file__))  +'/weights'

class TinyDesc(object):
	model = None
	PATCH_SIZE = 32

	def __init__(self, model_name):
		# load json and create model
		json_file = open(pretrained_path + '/' + model_name + '.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		loaded_model.load_weights(pretrained_path + '/' + model_name + '.h5')

		print("Loaded model from disk")
		 
		# evaluate loaded model on test data
		#loaded_model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

		self.model = loaded_model
