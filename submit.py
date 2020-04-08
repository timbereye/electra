import os
import shutil
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("eval_file_path", None, "evaluate file path")
flags.DEFINE_string("answer_model_path", None, "answer model path")
flags.DEFINE_string("qa_model_path", None, "qa model path")
flags.DEFINE_string("output_dir", None, "output_dir")


def check_and_mkdirs(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return True


def main():
    output_dir = FLAGS.output_dir
    qa_model_dir = FLAGS.qa_model_path
    answer_model_dir = FLAGS.answer_model_path
    check_and_mkdirs(output_dir)
    raw_data_dir = os.path.join(output_dir, "finetuning_data", "squad")
    check_and_mkdirs(raw_data_dir)
    eval_file_path = FLAGS.eval_file_path
    shutil.copy(eval_file_path, os.path.join(raw_data_dir, "dev.json"))

    qa_xargs = f"""python run_finetuning.py \
  --data-dir={output_dir} \
  --hparams '{{"model_size": "large", "task_names": ["squad"], "eval_batch_size": 8, "predict_batch_size": 8, "max_seq_length": 512, "learning_rate": 5e-5, "use_tfrecords_if_existing": false, "do_train": false, "do_eval": true, "pretrained_model_dir": {qa_model_dir}, "results_dir_name":"results_qa" }}'
"""
    os.system(qa_xargs)

    answer_xargs = f"""python run_finetuning.py \
  --data-dir={output_dir} \
  --hparams '{{"model_size": "large", "task_names": ["squad"], "eval_batch_size": 8, "predict_batch_size": 8, "max_seq_length": 512, "learning_rate": 5e-5, "use_tfrecords_if_existing": true, "do_train": false, "do_eval": true, "pretrained_model_dir": {answer_model_dir}, "results_dir_name":"results_answer"}}'
"""
    os.system(answer_xargs)


if __name__ == '__main__':
    main()
