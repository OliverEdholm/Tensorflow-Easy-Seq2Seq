# imports
import tensorflow as tf


# functions
def get_rnn_cell(rnn_cell_string):
    cell_string = rnn_cell_string.lower()

    if cell_string == 'gru':
        rnn_cell = tf.contrib.rnn.GRUCell
    elif cell_string == 'lstm':
        rnn_cell = tf.contrib.rnn.LSTMCell
    else:
        raise ValueError('The RNN cell {} isn\'t '
                         'implemented in this script.'.format(rnn_cell_string))

    return rnn_cell
