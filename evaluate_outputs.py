# imports
import sys
sys.path.append('src')

from config import DEFAULT_CHECKPOINT_PATH
from config import DEFAULT_DATA_SET_PATH
from data import DataSet
from predicting import Predicter

import logging

import tensorflow as tf

# setup
logging.basicConfig(level=logging.DEBUG)

# variables
flags = tf.app.flags
flags.DEFINE_string('eval_text', None, 'Text to get output from.')
flags.DEFINE_string('checkpoint_path', DEFAULT_CHECKPOINT_PATH,
                    'Path to checkpoint, if no file is given it tries to get '
                    'the latest checkpoint in that folder.')
flags.DEFINE_string('data_set_path', DEFAULT_DATA_SET_PATH,
                    'Path to dataset.')
FLAGS = flags.FLAGS


# functions
def main():
    data_set = DataSet.load(FLAGS.data_set_path)
    print(data_set.output_vocabulary.token_to_id)

    logging.info('building predicter')
    predicter = Predicter(data_set.output_vocabulary, FLAGS.checkpoint_path)

    logging.info('getting output')
    text = ' '.join(predicter.get_output(FLAGS.eval_text))
    print(text)


if __name__ == '__main__':
    main()
