# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code for serializing raw fine-tuning data into tfrecords"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import random

import dill
import numpy as np
import tensorflow.compat.v1 as tf

import configure_finetuning
from finetune import feature_spec
from util import utils


class Preprocessor(object):
    """Class for loading, preprocessing, and serializing fine-tuning datasets."""

    def __init__(self, config: configure_finetuning.FinetuningConfig, tasks, do_ensemble=False):
        self._config = config
        self._tasks = tasks
        self._name_to_task = {task.name: task for task in tasks}
        self.do_ensemble = do_ensemble

        self._feature_specs = feature_spec.get_shared_feature_specs(config)
        for task in tasks:
            if task.name == "squad" and self.do_ensemble:
                self._feature_specs += task.get_feature_specs(do_ensemble=self.do_ensemble)
            else:
                self._feature_specs += task.get_feature_specs()
        self._name_to_feature_config = {
            spec.name: spec.get_parsing_spec()
            for spec in self._feature_specs
        }
        assert len(self._name_to_feature_config) == len(self._feature_specs)

    def prepare_train(self, sub=None):
        return self._serialize_dataset(self._tasks, True, "train", sub)

    def prepare_predict(self, tasks, split, prepare_ensemble=False):  # 在生成train.json/dev.json的feature时保存unique id信息，用于流式生成logits的顺序控制
        return self._serialize_dataset(tasks, False, split, prepare_ensemble=prepare_ensemble)

    def _serialize_dataset(self, tasks, is_training, split, sub=None, prepare_ensemble=False):
        """Write out the dataset as tfrecords."""
        dataset_name = "_".join(sorted([task.name for task in tasks]))
        dataset_name += "_" + split
        dataset_prefix = os.path.join(
            self._config.preprocessed_data_dir(str(sub) if sub else ""), dataset_name)
        if self.do_ensemble:  # ensemble模型需要logits信息，重新保存
            dataset_prefix = dataset_prefix + "_ensemble"
        tfrecords_path = dataset_prefix + ".tfrecord"
        metadata_path = dataset_prefix + ".metadata"
        batch_size = (self._config.train_batch_size if is_training else
                      self._config.eval_batch_size)

        utils.log("Loading dataset", dataset_name)
        n_examples = None
        if (self._config.use_tfrecords_if_existing and
                tf.io.gfile.exists(metadata_path)):
            n_examples = utils.load_json(metadata_path)["n_examples"]

        if n_examples is None:
            utils.log("Existing tfrecords not found so creating")
            examples = []
            for task in tasks:
                task_examples = task.get_examples(split, sub)
                examples += task_examples
            if is_training:
                random.shuffle(examples)
            utils.mkdir(tfrecords_path.rsplit("/", 1)[0])
            n_examples = self.serialize_examples(
                examples, is_training, tfrecords_path, batch_size, split=split, prepare_ensemble=prepare_ensemble)
            utils.write_json({"n_examples": n_examples}, metadata_path)

        input_fn = self._input_fn_builder(tfrecords_path, is_training)
        if is_training:
            steps = int(n_examples // batch_size * self._config.num_train_epochs)
        else:
            steps = n_examples // batch_size

        return input_fn, steps

    def serialize_examples(self, examples, is_training, output_file, batch_size, split="train", prepare_ensemble=False):
        """Convert a set of `InputExample`s to a TFRecord file."""
        _logits_fps = []
        if self.do_ensemble:  # 加载子模型生成的logits
            for i in range(self._config.ensemble_k):
                _logits_fps.append(tf.gfile.Open(self._config.logits_tmp(split + str(i)), 'rb'))

        unique_ids_file = self._config.unique_ids_tmp(split)
        unique_ids = None
        if prepare_ensemble and not tf.gfile.Exists(unique_ids_file):
            unique_ids = []

        n_examples = 0
        with tf.io.TFRecordWriter(output_file) as writer:
            for (ex_index, example) in enumerate(examples):
                if ex_index % 2000 == 0:
                    utils.log("Writing example {:} of {:}".format(
                        ex_index, len(examples)))
                for tf_example in self._example_to_tf_example(
                        example, is_training, log=self._config.log_examples and ex_index < 1, logits_fps=_logits_fps, unique_ids=unique_ids):
                    writer.write(tf_example.SerializeToString())
                    n_examples += 1
            # add padding so the dataset is a multiple of batch_size
            while n_examples % batch_size != 0:
                writer.write(self._make_tf_example(task_id=len(self._config.task_names))
                             .SerializeToString())
                n_examples += 1

        if prepare_ensemble and unique_ids:
            with tf.gfile.Open(unique_ids_file, 'wb') as fp:
                dill.dump(unique_ids, fp)

        if self.do_ensemble and _logits_fps:
            for fp in _logits_fps:
                fp.close()

        return n_examples

    def _example_to_tf_example(self, example, is_training, log=False, logits_fps=None, unique_ids=None):  # 流式读写pkl避免OOM
        task_name = example.task_name
        examples = self._name_to_task[example.task_name].featurize(
            example, is_training, log)
        if not isinstance(examples, list):
            examples = [examples]
        for example in examples:
            unique_id = example[task_name + "_eid"]
            if unique_ids is not None:
                unique_ids.append(unique_id)
            if self.do_ensemble:
                for i in range(self._config.ensemble_k):
                    logits = dill.load(logits_fps[i])
                    assert unique_id in logits
                    example_logits_info = logits[unique_id]
                    example.update(
                        {task_name + "_start_logits" + "_" + str(i): example_logits_info[0],
                         task_name + "_end_logits" + "_" + str(i): example_logits_info[1],
                         task_name + "_answerable_logit" + "_" + str(i): example_logits_info[2]}
                    )
            yield self._make_tf_example(**example)

    def _make_tf_example(self, **kwargs):
        """Make a tf.train.Example from the provided features."""
        for k in kwargs:
            if k not in self._name_to_feature_config:
                raise ValueError("Unknown feature", k)
        features = collections.OrderedDict()
        for spec in self._feature_specs:
            if spec.name in kwargs:
                values = kwargs[spec.name]
            else:
                values = spec.get_default_values()
            if (isinstance(values, int) or isinstance(values, bool) or
                    isinstance(values, float) or isinstance(values, np.float32) or
                    (isinstance(values, np.ndarray) and values.size == 1)):
                values = [values]
            if spec.is_int_feature:
                feature = tf.train.Feature(int64_list=tf.train.Int64List(
                    value=list(values)))
            else:
                feature = tf.train.Feature(float_list=tf.train.FloatList(
                    value=list(values)))
            features[spec.name] = feature
        return tf.train.Example(features=tf.train.Features(feature=features))

    def _input_fn_builder(self, input_file, is_training):
        """Creates an `input_fn` closure to be passed to TPUEstimator."""

        def input_fn(params):
            """The actual input function."""
            d = tf.data.TFRecordDataset(input_file)
            if is_training:
                d = d.repeat()
                d = d.shuffle(buffer_size=100)
            print("params:", params)
            return d.apply(
                tf.data.experimental.map_and_batch(
                    self._decode_tfrecord,
                    batch_size=params["batch_size"],
                    drop_remainder=True))

        return input_fn

    def _decode_tfrecord(self, record):
        """Decodes a record to a TensorFlow example."""
        example = tf.io.parse_single_example(record, self._name_to_feature_config)
        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name, tensor in example.items():
            if tensor.dtype == tf.int64:
                example[name] = tf.cast(tensor, tf.int32)
            else:
                example[name] = tensor
        return example
