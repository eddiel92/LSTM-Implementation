import os
import time
import itertools
import tensorflow as tf
import model
import hyperparameters
import metrics
import inputs
from models.dual_encoder import dual_encoder_model

tf.flags.DEFINE_string("input_dir", "scripts/data", "Directory containing input data files 'train.tfrecords' and 'validation.tfrecords'")
tf.flags.DEFINE_string("model_dir", None, "Directory to store model checkpoints (defaults to ./runs)")
tf.flags.DEFINE_integer("loglevel", 20, "Tensorflow log level")
tf.flags.DEFINE_integer("num_epochs", None, "Number of training Epochs. Defaults to indefinite.")
tf.flags.DEFINE_integer("eval_every", 2000, "Evaluate after this many train steps")
FLAGS = tf.flags.FLAGS

TIMESTAMP = int(time.time())

if FLAGS.model_dir:
  MODEL_DIR = FLAGS.model_dir
else:
  MODEL_DIR = os.path.abspath(os.path.join("./runs", str(TIMESTAMP)))

TRAIN_FILE = os.path.abspath(os.path.join(FLAGS.input_dir, "train.tfrecords"))
VALIDATION_FILE = os.path.abspath(os.path.join(FLAGS.input_dir, "validation.tfrecords"))

tf.logging.set_verbosity(FLAGS.loglevel)

def main(unused_argv):

  # Replace MODEL_DIR with the folder current run to resume training from a set of hyperparameters
  # MODEL_DIR = '/Users/eduardolitonjua/Desktop/Retrieval-System/runs/1472130056' 
  hparams = hyperparameters.create_hparams()

  model_fn = model.create_model_fn(
    hparams,
    model_impl=dual_encoder_model)

  estimator = tf.contrib.learn.Estimator(
    model_fn=model_fn,
    model_dir=MODEL_DIR,
    config=tf.contrib.learn.RunConfig())

  input_fn_train = inputs.create_input_fn(
    mode=tf.contrib.learn.ModeKeys.TRAIN,
    input_files=[TRAIN_FILE],
    batch_size=hparams.batch_size,
    num_epochs=FLAGS.num_epochs)

  input_fn_eval = inputs.create_input_fn(
    mode=tf.contrib.learn.ModeKeys.EVAL,
    input_files=[VALIDATION_FILE],
    batch_size=hparams.eval_batch_size,
    num_epochs=1)

  eval_metrics = metrics.create_evaluation_metrics()

  class EvaluationMonitor(tf.contrib.learn.monitors.EveryN):
    def every_n_step_end(self, step, outputs):
      self._estimator.evaluate(
        input_fn=input_fn_eval,
        metrics=eval_metrics,
        steps=None)

  eval_monitor = EvaluationMonitor(every_n_steps=FLAGS.eval_every, first_n_steps=-1)
  estimator.fit(input_fn=input_fn_train, steps=None, monitors=[eval_monitor])

if __name__ == "__main__":
  tf.app.run()
