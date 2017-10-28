from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from datetime import datetime
import argparse
import sys
import tensorflow as tf
import captcha_model as captcha

FLAGS = None


def run_train():
  """Train CAPTCHA for a number of steps."""

  with tf.Graph().as_default():
    images, labels = captcha.inputs(train=True, batch_size=FLAGS.batch_size)
    test_images,test_labels = captcha.inputs(train=False, batch_size=FLAGS.batch_size)


    logits = captcha.inference(images, keep_prob=0.75,is_training=True)
    test_logits = captcha.inference(test_images, keep_prob=1,is_training=False)

    loss = captcha.loss(logits, labels)

    test_correct = captcha.evaluation(test_logits, test_labels)#test
    correct = captcha.evaluation(logits, labels)#train

    train_precision = correct/FLAGS.batch_size
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('train_precision', train_precision)
    tf.summary.image('images',images,10)
    summary = tf.summary.merge_all()
    train_op = captcha.training(loss)
    saver = tf.train.Saver()

#    init_op = tf.group(tf.global_variables_initializer(),
#                       tf.local_variables_initializer())

    sess = tf.Session()
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
#    sess.run(init_op)
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
      step = 120140
      while not coord.should_stop():
        start_time = time.time()
        _, loss_value,test_value, train_value = sess.run([train_op, loss, test_correct,correct])
        summary_str = sess.run(summary)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

        duration = time.time() - start_time
        step += 1
        if step % 10 == 0:
          print('>> Step %d run_train: loss = %.2f, test = %.2f, train = %.2f (%.3f sec)'
                % (step, loss_value,test_value, train_value, duration))
          #-------------------------------

        if step % 100 == 0:
          print('>> %s Saving in %s' % (datetime.now(), FLAGS.checkpoint))
          saver.save(sess, FLAGS.checkpoint, global_step=step)
          print(images.shape.as_list(),labels.shape.as_list())

        if step>2000000:
          break
    except KeyboardInterrupt:
      print('INTERRUPTED')
      coord.request_stop()
    except Exception as e:

      coord.request_stop(e)
    finally:
      saver.save(sess, FLAGS.checkpoint, global_step=step)
      print('Model saved in file :%s'%FLAGS.checkpoint)

      coord.request_stop()
      coord.join(threads)
    sess.close()



def main(_):
#  if tf.gfile.Exists(FLAGS.train_dir):
#    tf.gfile.DeleteRecursively(FLAGS.train_dir)
#  tf.gfile.MakeDirs(FLAGS.train_dir)
  run_train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--batch_size',
      type=int,
      default=60,
      help='Batch size.'
  )
  parser.add_argument(
      '--train_dir',
      type=str,
      default='captcha_train',
      help='Directory where to write event logs.'
  )
  parser.add_argument(
      '--checkpoint_dir',
      type=str,
      default='./checkpoint',
      help='Directory where to restore checkpoint.'
  )
  parser.add_argument(
      '--checkpoint',
      type=str,
      default='checkpoint/model.ckpt',
      help='Directory where to write checkpoint.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
