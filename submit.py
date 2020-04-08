import os
import shutil
import tensorflow as tf
from external_verifier import ev

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("eval_input_file", None, "evaluate input file path")
flags.DEFINE_string("answer_model_path", None, "answer model path")
flags.DEFINE_string("qa_model_path", None, "qa model path")
flags.DEFINE_string("output_dir", None, "output_dir")
flags.DEFINE_string("output_file", None, "output predictions file")


def check_and_mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return True


def main():
    output_dir = os.path.abspath(FLAGS.output_dir)
    qa_model_dir = FLAGS.qa_model_path
    answer_model_dir = FLAGS.answer_model_path
    output_file = FLAGS.output_file
    check_and_mkdirs(output_dir)
    raw_data_dir = os.path.join(output_dir, "finetuning_data", "squad")
    check_and_mkdirs(raw_data_dir)
    eval_file_path = FLAGS.eval_input_file
    shutil.copy(eval_file_path, os.path.join(raw_data_dir, "dev.json"))

    qa_xargs = f"""python run_finetuning.py \
  --data-dir="{output_dir}" \
  --hparams '{{"model_size": "large", "task_names": ["squad"], "eval_batch_size": 8, "predict_batch_size": 8, "max_seq_length": 512, "learning_rate": 5e-5, "use_tfrecords_if_existing": false, "do_train": false, "do_eval": true, "pretrained_model_dir": "{qa_model_dir}", "results_dir_name":"results_qa" }}'
"""
    # os.system(qa_xargs)

    answer_xargs = f"""python run_finetuning.py \
  --data-dir="{output_dir}" \
  --hparams '{{"model_size": "large", "task_names": ["squad"], "eval_batch_size": 8, "predict_batch_size": 8, "max_seq_length": 512, "learning_rate": 5e-5, "use_tfrecords_if_existing": true, "do_train": false, "do_eval": true, "pretrained_model_dir": "{answer_model_dir}", "results_dir_name":"results_answer"}}'
"""
    # os.system(answer_xargs)

    qa_preds_file = os.path.join(output_dir, "results_qa", "squad_qa", "squad_preds.json")
    qa_eval_result_file = os.path.join(output_dir, "results_qa", "squad_qa", "squad_eval.json")
    qa_null_odds_file = os.path.join(output_dir, "results_qa", "squad_qa", "squad_null_odds.json")
    answer_null_odds_file = os.path.join(output_dir, "results_answer", "squad_qa", "squad_null_odds.json")

    ev(eval_file_path, qa_preds_file, qa_null_odds_file, qa_eval_result_file, answer_null_odds_file, output_file)


if __name__ == '__main__':
    main()
