# imports
import os
import logging
from shutil import rmtree


# functions
def ask_yes_or_no(question):
    answer = ''
    while answer not in ['y', 'n']:
        inp = input('{} y / n: '.format(question))

        answer = inp.lower()[0]

    return answer == 'y'


def fix_path(path, is_folder=True):
    create_folder = False

    if os.path.exists(path):
        if ask_yes_or_no('There already exists something at {}, '
                         'do you want to replace it?'.format(path)):
            logging.debug('removing at {}'.format(path))
            if is_folder:
                rmtree(path)
            else:
                os.remove(path)

            create_folder = True
    else:
        create_folder = True

    if create_folder and is_folder:
        logging.debug('creating at {}'.format(path))
        os.makedirs(path)

    return create_folder


def get_text_file(file_path):
    with open(file_path, 'r') as text_file:
        return text_file.readlines()


def get_sorted_buckets(buckets):
    buckets = list(buckets)  # in case of being tuple

    return sorted(buckets, key=lambda b: b[0])


def get_buckets(buckets_string):
    buckets = eval(buckets_string)

    return get_sorted_buckets(buckets)
