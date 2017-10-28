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
Batch_size = 32

def one_hot_to_texts(recog_result):
  texts = []
  for i in range(recog_result.shape[0]):
    index = recog_result[i]
    texts.append(''.join([CHAR_SETS[i] for i in index]))
  return texts


def input_data(image_dir):
  if not gfile.Exists(image_dir):
    print(">> Image director '" + image_dir + "' not found.")
    return None
  extensions = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG']
  print(">> Looking for images in '" + image_dir + "'")
  file_list = []
  for extension in extensions:
    file_glob = os.path.join(image_dir, '*.' + extension)
    file_list.extend(gfile.Glob(file_glob))
  if not file_list:
    print(">> No files found in '" + image_dir + "'")
    return None
  all_files = len(file_list)
  images = np.zeros([all_files, IMAGE_HEIGHT*IMAGE_WIDTH], dtype='float32')
  files = []
  i = 0
  for file_name in file_list:
    image = Image.open(file_name)
    image_gray = image.convert('L')
    
    image_resize = image_gray.resize(size=(IMAGE_WIDTH,IMAGE_HEIGHT))
    image.close()
    input_img = np.array(image_resize, dtype='float32')
    input_img = input_img.flatten()/127.5 - 1    
    images[i,:] = input_img
    base_name = os.path.basename(file_name)
    files.append(base_name)
    i += 1
  return images, files


def run_predict():
  with tf.Graph().as_default():
    input_images, input_filenames = input_data(FLAGS.captcha_dir)#得到文件夹内所有照片和文件名
    epoches = len(input_images)//Batch_size 
    offset = len(input_images) - (epoches-1)*Batch_size
    images = tf.placeholder(tf.float32,[Batch_size,IMAGE_HEIGHT*IMAGE_WIDTH],name ='input')
    logits = captcha.inference(images, keep_prob=1,is_training=True)
    result = captcha.output(logits)
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
    for each in range(epoches):
        feed_dict = input_images[each*Batch_size:min((each+1)*Batch_size,len(input_images))]
        
        recog_result = sess.run(result,feed_dict={images:feed_dict})
       
        text = one_hot_to_texts(recog_result)
        total_count = len(feed_dict)
        true_count = 0.
        for i in range(total_count):
          print('image ' + input_filenames[i+each*Batch_size] + " recognize ----> '" + text[i] + "'")
          with open('recognize.txt','a') as f:
            f.write('image ' + input_filenames[i+each*Batch_size] + 'recognize ----> ' + text[i] +'\n')
          if text[i] in input_filenames[i+each*Batch_size]:
            true_count += 1
        precision = true_count / total_count
        
        print('%s epoch: %d ,true/total: %d/%d recognize @ = %.3f'
                        %(datetime.now(), each+1, true_count, total_count, precision))
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
