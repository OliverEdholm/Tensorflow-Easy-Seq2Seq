# imports
import sys
sys.path.append('src')

from config import DEFAULT_BUCKETS_STRING
from config import DEFAULT_DATA_SET_PATH
from config import DEFAULT_LAYER_SIZE
from config import DEFAULT_N_LAYERS
from config import DEFAULT_MAX_GRADIENT_NORM
from config import DEFAULT_BATCH_SIZE
from config import DEFAULT_LEARNING_RATE
from config import DEFAULT_LEARNING_RATE_DECAY_FACTOR
from config import DEFAULT_RNN_CELL
from config import DEFAULT_N_SAMPLES
from config import DEFAULT_FORWARD_ONLY
from config import DEFAULT_CHECKPOINT_INTERVAL
from config import DEFAULT_EVAL_INTERVAL
from config import DEFAULT_LOSS_CHECK_INTERVAL
from config import DEFAULT_CHECKPOINT_PATH
from config import DEFAULT_N_EVAL_BATCHES
from config import DEFAULT_N_STEPS
from model_utils import get_rnn_cell
from utils import get_buckets
from utils import fix_path
from model import Seq2Seq
from data import DataSet

import os
import logging
from six.moves import xrange

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# setup
logging.basicConfig(level=logging.DEBUG)

# variables
flags = tf.app.flags
flags.DEFINE_string('buckets',
                    DEFAULT_BUCKETS_STRING,
                    'Buckets written as a python list with tuples in it.')
flags.DEFINE_string('data_set_path', DEFAULT_DATA_SET_PATH,
                    'Path to saving destination of dataset.')
flags.DEFINE_integer('layer_size', DEFAULT_LAYER_SIZE,
                     'Layer size of rnn cell.')
flags.DEFINE_integer('n_layers', DEFAULT_N_LAYERS,
                     'Number of layers of RNN cell.')
flags.DEFINE_float('max_gradient_norm', DEFAULT_MAX_GRADIENT_NORM,
                   'Max gradient norm.')
flags.DEFINE_integer('batch_size', DEFAULT_BATCH_SIZE, 'Batch size.')
flags.DEFINE_float('learning_rate', DEFAULT_LEARNING_RATE, 'Learning rate.')
flags.DEFINE_float('learning_rate_decay_factor',
                   DEFAULT_LEARNING_RATE_DECAY_FACTOR,
                   'Learning rate decay factor is how big of a percentage of'
                   'the current learning rate that should be kept.')
flags.DEFINE_string('rnn_cell', DEFAULT_RNN_CELL,
                    'Which RNN cell that should be used, either GRU or LSTM.')
flags.DEFINE_integer('n_samples', DEFAULT_N_SAMPLES,
                     'Number of words from softmax to be calculated.')
flags.DEFINE_boolean('forward_only', DEFAULT_FORWARD_ONLY,
                     'Forward only.')
flags.DEFINE_integer('checkpoint_interval', DEFAULT_CHECKPOINT_INTERVAL,
                     'How many steps should it wait to create a checkpoint.')
flags.DEFINE_integer('loss_check_interval', DEFAULT_LOSS_CHECK_INTERVAL,
                     'How many steps should it wait to check if learning rate'
                     ' have to be lower.')
flags.DEFINE_integer('eval_interval', DEFAULT_EVAL_INTERVAL,
                     'How many steps should it wait to evaluate.')
flags.DEFINE_string('checkpoint_path', DEFAULT_CHECKPOINT_PATH,
                    'Path to save checkpoint on.')
flags.DEFINE_integer('n_eval_batches', DEFAULT_N_EVAL_BATCHES,
                     'Path to save checkpoint on.')
flags.DEFINE_integer('n_steps', DEFAULT_N_STEPS,
                     'Number of steps to train on.')
FLAGS = flags.FLAGS


# classes
class BucketChooser:
    def __init__(self, data_set, use_train_data):
        self.buckets_scale = self._get_buckets_scale(data_set, use_train_data)

    def _get_correct_data(self, data_set, use_train_data):
        if use_train_data:
            return data_set.train_data_set
        else:
            return data_set.eval_data_set

    def _get_buckets_scale(self, data_set, use_train_data):
        data = self._get_correct_data(data_set, use_train_data)

        n_buckets = len(data_set.buckets)

        bucket_sizes = [len(data[i])
                        for i in xrange(n_buckets)]
        total_size = sum(bucket_sizes)

        return [float(sum(bucket_sizes[:i + 1]) / total_size)
                for i in xrange(n_buckets)]

    def get_bucket_id(self):
        random_unit_interval = np.random.random_sample()

        return min([i for i in xrange(len(self.buckets_scale))
                    if self.buckets_scale[i] > random_unit_interval])


# functions
def get_data_set():
    data_set = DataSet.load(FLAGS.data_set_path)

    return data_set


def get_model(input_vocab_size, output_vocab_size):
    return Seq2Seq(input_vocab_size, output_vocab_size,
                   get_buckets(FLAGS.buckets), FLAGS.layer_size,
                   FLAGS.n_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
                   FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
                   get_rnn_cell(FLAGS.rnn_cell), FLAGS.n_samples,
                   FLAGS.forward_only)


def fix_paths():
    fix_path(FLAGS.checkpoint_path)


def evaluate_model(sess, model, bucket_chooser, data_set,
                   n_batches=DEFAULT_N_EVAL_BATCHES):
    losses = []
    for _ in tqdm(xrange(n_batches)):
        bucket_id = bucket_chooser.get_bucket_id()

        batches = data_set.get_random_batch(FLAGS.batch_size, bucket_id)
        batch_encoder_inputs = batches[0]
        batch_decoder_inputs = batches[1]
        batch_target_weights = batches[2]

        _, loss, _ = model.step(
            sess, batch_encoder_inputs, batch_decoder_inputs,
            batch_target_weights, bucket_id, True)

        losses.append(loss)

    return np.mean(losses)


def main():
    fix_paths()

    data_set = get_data_set()

    train_bucket_chooser = BucketChooser(data_set, True)
    eval_bucket_chooser = BucketChooser(data_set, False)

    model = get_model(data_set.input_vocabulary_size,
                      data_set.output_vocabulary_size)

    loss = 0
    previous_losses = []

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        logging.debug('initializing')
        sess.run(init)

        logging.debug('training')
        for step in xrange(1, FLAGS.n_steps + 1):
            bucket_id = train_bucket_chooser.get_bucket_id()

            batches = data_set.get_random_batch(FLAGS.batch_size, bucket_id)
            batch_encoder_inputs = batches[0]
            batch_decoder_inputs = batches[1]
            batch_target_weights = batches[2]

            _, step_loss, _ = model.step(
                sess, batch_encoder_inputs, batch_decoder_inputs,
                batch_target_weights, bucket_id, False)

            loss += step_loss / FLAGS.eval_interval

            if step % FLAGS.checkpoint_interval == 0:
                print('step: {}, saving checkpoint'.format(step))
                checkpoint_path = os.path.join(FLAGS.checkpoint_path,
                                               'seq2seq')
                model.save_checkpoint(sess, checkpoint_path)

            if step % FLAGS.loss_check_interval == 0:
                logging.debug('checking loss')
                print('step: {}, loss: {}'.format(step, loss))

                previous_losses.append(loss)

                if len(previous_losses) > 2 and \
                   loss > max(previous_losses[-3:]):
                    logging.debug('decreasing learning rate')
                    sess.run(model.learning_rate_decay_operation)

                loss = 0

            if step % FLAGS.eval_interval == 0:
                logging.debug('evaluating model')
                eval_loss = evaluate_model(sess, model, eval_bucket_chooser,
                                           data_set)
                print('step: {}, eval loss: {}'.format(step, eval_loss))


if __name__ == '__main__':
    main()
