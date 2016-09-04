import functools
import tensorflow as tf
import numpy as np
import array
import os
import csv
import itertools


tf.flags.DEFINE_integer(
  "min", 5, "Minimum frequency of words in the vocabulary")

tf.flags.DEFINE_string(
  "input_dir", os.path.abspath("./data"),
  "Input directory containing original CSV data files (default = './data')")

tf.flags.DEFINE_integer("max", 200, "maximum sentence length")

tf.flags.DEFINE_string(
  "output_dir", os.path.abspath("./data"),
  "Output directory (default = './data')")

FLAGS = tf.flags.FLAGS

tr = os.path.join(FLAGS.input_dir, "train.csv")
val = os.path.join(FLAGS.input_dir, "valid.csv")
tp = os.path.join(FLAGS.input_dir, "test.csv")

def tokenizer_fn(iterator):
  return (x.split(" ") for x in iterator)

def create_csv_iter(filename):
  with open(filename) as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
      yield row


def create_vocab(input_iter, min_frequency):
  vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
      FLAGS.max,
      min_frequency=min_frequency,
      tokenizer_fn=tokenizer_fn)
  vocab_processor.fit(input_iter)
  return vocab_processor


def transform_sentence(sequence, vocab_processor):
  return next(vocab_processor.transform([sequence])).tolist()


def create_text_sequence_feature(fl, sentence, sentence_len, vocab):
  sentence_transformed = transform_sentence(sentence, vocab)
  for word_id in sentence_transformed:
    fl.feature.add().int64_list.value.extend([word_id])
  return fl


def create_example_train(row, vocab):
  context, utterance, label = row
  context_transformed = transform_sentence(context, vocab)
  utterance_transformed = transform_sentence(utterance, vocab)
  context_len = len(next(vocab._tokenizer([context])))
  utterance_len = len(next(vocab._tokenizer([utterance])))
  label = int(float(label))

  example = tf.train.Example()
  example.features.feature["context"].int64_list.value.extend(context_transformed)
  example.features.feature["utterance"].int64_list.value.extend(utterance_transformed)
  example.features.feature["context_len"].int64_list.value.extend([context_len])
  example.features.feature["utterance_len"].int64_list.value.extend([utterance_len])
  example.features.feature["label"].int64_list.value.extend([label])
  return example


def create_example_test(row, vocab):
  context, utterance = row[:2]
  distractors = row[2:]
  context_len = len(next(vocab._tokenizer([context])))
  utterance_len = len(next(vocab._tokenizer([utterance])))
  context_transformed = transform_sentence(context, vocab)
  utterance_transformed = transform_sentence(utterance, vocab)

  example = tf.train.Example()
  example.features.feature["context"].int64_list.value.extend(context_transformed)
  example.features.feature["utterance"].int64_list.value.extend(utterance_transformed)
  example.features.feature["context_len"].int64_list.value.extend([context_len])
  example.features.feature["utterance_len"].int64_list.value.extend([utterance_len])

  for i, distractor in enumerate(distractors):
    dis_key = "distractor_{}".format(i)
    dis_len_key = "distractor_{}_len".format(i)

    dis_len = len(next(vocab._tokenizer([distractor])))
    example.features.feature[dis_len_key].int64_list.value.extend([dis_len])

    dis_transformed = transform_sentence(distractor, vocab)
    example.features.feature[dis_key].int64_list.value.extend(dis_transformed)
  return example


def create_tfrecords_file(input_filename, output_filename, example_fn):

  writer = tf.python_io.TFRecordWriter(output_filename)
  print("Creating TFRecords file at {}...".format(output_filename))
  for i, row in enumerate(create_csv_iter(input_filename)):
    x = example_fn(row)
    writer.write(x.SerializeToString())
  writer.close()
  print("Wrote to {}".format(output_filename))


def write_vocabulary(vocab_processor, outfile):

  vocab_size = len(vocab_processor.vocabulary_)
  with open(outfile, "w") as vocabfile:
    for id in range(vocab_size):
      word =  vocab_processor.vocabulary_._reverse_mapping[id]
      vocabfile.write(word + "\n")
  print("Saved vocabulary to {}".format(outfile))


if __name__ == "__main__":
  print("Creating vocabulary...")
  input_iter = create_csv_iter(tr)
  input_iter = (x[0] + " " + x[1] for x in input_iter)
  vocab = create_vocab(input_iter, min_frequency=FLAGS.min)
  print("Total vocabulary size: {}".format(len(vocab.vocabulary_)))

  write_vocabulary(
    vocab, os.path.join(FLAGS.output_dir, "vocabulary.txt"))

  vocab.save(os.path.join(FLAGS.output_dir, "vocab_processor.bin"))

  create_tfrecords_file(
      input_filename=val,
      output_filename=os.path.join(FLAGS.output_dir, "validation.tfrecords"),
      example_fn=functools.partial(create_example_test, vocab=vocab))

  create_tfrecords_file(
      input_filename=tp,
      output_filename=os.path.join(FLAGS.output_dir, "test.tfrecords"),
      example_fn=functools.partial(create_example_test, vocab=vocab))

  create_tfrecords_file(
      input_filename=tr,
      output_filename=os.path.join(FLAGS.output_dir, "train.tfrecords"),
      example_fn=functools.partial(create_example_train, vocab=vocab))
