import sugartensor as sg
import numpy as np
import csv
import tensorflow as tf

@sg.sg_producer_func
def _load_data(src_list):
    labels, data = src_list

    labels = np.array(labels, np.float)
    data = np.array(data, np.float)

    return labels, data

class Surv(object):

    def __init__(self):
        data, labels = [], []
        with open('data/Housing.csv') as csv_file:
            reader = csv.reader(csv_file)

            i = 0
            for row in reader:
                if i != 0:
                    labels.append(row[1])
                    cols = []
                    for col in row[2:]:
                        if col == 'yes':
                            col = '1'
                        elif col == 'no':
                            col = '0'
                        cols.append(col)
                    data.append(cols)
                i += 1

        label_t = tf.convert_to_tensor(labels)
        data_t = tf.convert_to_tensor(data)

        label_q, data_q\
            = tf.train.slice_input_producer([label_t, data_t], shuffle=True)

        label_q, data_q = _load_data(source=[label_q, data_q],
                                     dtypes=[sg.sg_intx, sg.sg_intx],
                                     capacity=256, num_threads=64)

        batch_queue = tf.train.batch([label_q, data_q], batch_size=32,
                                     shapes=[(), (11)],
                                     num_threads=64, capacity=32*32,
                                     dynamic_pad=False)

        self.label, self.data = batch_queue
        self.num_batch = len(labels) // 32

        sg.sg_info('%s set loaded.(total data=%d, total batch=%d)'
                   % ('train', len(labels), self.num_batch))
