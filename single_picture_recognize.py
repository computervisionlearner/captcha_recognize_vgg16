#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 22:38:23 2017

@author: no1
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os.path
from datetime import datetime
from PIL import Image
import numpy as np

import tensorflow as tf
from tensorflow.python.platform import gfile
import captcha_model as captcha

import config

IMAGE_WIDTH = config.IMAGE_WIDTH
IMAGE_HEIGHT = config.IMAGE_HEIGHT

CHAR_SETS = config.CHAR_SETS
CLASSES_NUM = config.CLASSES_NUM
CHARS_NUM = config.CHARS_NUM

FLAGS = None


def one_hot_to_texts(recog_result):
  texts = []
  for i in range(recog_result.shape[0]):
    index = recog_result[i]
    texts.append(''.join([CHAR_SETS[i] for i in index]))
  return texts



def run_predict():
  with tf.Graph().as_default():

    #------------------------------------------
    image_placeholder = tf.placeholder(dtype=tf.float32,shape=[1,IMAGE_HEIGHT*IMAGE_WIDTH])
    logits = captcha.inference(image_placeholder, keep_prob=1,is_training=True)
    result = captcha.output(logits)
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
    while True:

      path = input('please input picture path: ')
      image = Image.open(path)
      image_gray = image.convert('L')
      image_resize = image_gray.resize(size=(IMAGE_WIDTH,IMAGE_HEIGHT))
      image.close()
      input_img = np.array(image_resize, dtype='float32')
      input_img = input_img.flatten()/127.5 - 1
      input_img = np.expand_dims(input_img,0)
      recog_result = sess.run(result,feed_dict={image_placeholder:input_img})
      text = one_hot_to_texts(recog_result)
      print(text)
      _exit = input('press "q" to exit or others to continue: ')

      if _exit == 'q':
          break
    sess.close()

def main(_):
  run_predict()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--checkpoint_dir',
      type=str,
      default='checkpoint',
      help='Directory where to restore checkpoint.'
  )
  parser.add_argument(
      '--captcha_dir',
      type=str,
      default='data/test_data',
      help='Directory where to get captcha images.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
