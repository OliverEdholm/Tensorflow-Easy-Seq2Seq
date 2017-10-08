# imports
import sys
sys.path.append('src')

from config import DEFAULT_INPUTS_FILE_PATH
from config import DEFAULT_OUTPUTS_FILE_PATH
from config import DEFAULT_DATA_SET_PATH
from config import DEFAULT_BUCKETS_STRING
from config import DEFAULT_INPUT_VOCAB_SIZE
from config import DEFAULT_OUTPUT_VOCAB_SIZE
from data import DataSet
from utils import get_buckets
from utils import fix_path

import logging

import tensorflow as tf

# setup
logging.basicConfig(level=logging.DEBUG)

# variables
flags = tf.app.flags
flags.DEFINE_string('inputs_file_path', DEFAULT_INPUTS_FILE_PATH,
                    'Path to file with inputs.')
flags.DEFINE_string('outputs_file_path', DEFAULT_OUTPUTS_FILE_PATH,
                    'Path to file with outputs.')
flags.DEFINE_string('data_set_path', DEFAULT_DATA_SET_PATH,
                    'Path to saving destination of dataset.')
flags.DEFINE_string('buckets',
                    DEFAULT_BUCKETS_STRING,
                    'Buckets written as a python list with tuples in it.')
flags.DEFINE_integer('input_vocab_size', DEFAULT_INPUT_VOCAB_SIZE,
                     'Max vocab size of inputs.')
flags.DEFINE_integer('output_vocab_size', DEFAULT_OUTPUT_VOCAB_SIZE,
                     'Max vocab size of outputs.')
FLAGS = flags.FLAGS


# functions
def get_data_set():
    data_set = DataSet(FLAGS.inputs_file_path, FLAGS.outputs_file_path,
                       FLAGS.input_vocab_size, FLAGS.output_vocab_size,
                       get_buckets(FLAGS.buckets))

    return data_set


def main():
    if fix_path(FLAGS.data_set_path, is_folder=False):
        data_set = get_data_set()

        data_set.save(FLAGS.data_set_path)
    else:
        logging.debug('didn\'t create dataset')


if __name__ == '__main__':
    main()
