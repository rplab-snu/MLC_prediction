import tensorflow as tf
import sugartensor as sg
from input_data import Surv
from model import MLP

sg.sg_verbosity(10)
batch_size = 16

data = Surv()

x = data.data
y = data.label

model = MLP()

logit = model.inference(x)

loss = tf.reduce_mean(logit.sg_mse(target=tf.cast(y, tf.float32)))

sg.sg_train(loss=loss, ep_size=data.num_batch, log_interval=10, save_dir='log')