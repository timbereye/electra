import os
import shutil
import argparse


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--eval-file', required=True, help="eval file")
    parser.add_argument('--data-dir', required=True, help="data_dir")
    parser.add_argument('--output-file', required=True, help="final predictions")
    args = parser.parse_args()

    shutil.copy(args.eval_file, os.path.join(args.data_dir, 'finetuning_data/squad/dev.json'))

    command1 = """
    python3 run_finetuning_atrlp.py   --data-dir=%s --model-name=atrlp8876_   --hparams '{"model_size": "large", "task_names": ["squad"], "use_tpu": false, "eval_batch_size": 16, "predict_batch_size": 16, "max_seq_length": 512, "use_tfrecords_if_existing": false, "num_trials": 1, "do_train": false, "do_eval": true}'
""" % args.data_dir

    os.system(command1)

    shutil.copy(os.path.join(args.data_dir, 'models/atrlp8876_/results/squad_qa/eval_all_nbest.pkl'),
                './data/dev_all_nbest.pkl')
    shutil.copy(os.path.join(args.data_dir, 'models/atrlp8876_/results/squad_qa/squad_eval.json'),
                './data/squad_eval.json')
    shutil.copy(os.path.join(args.data_dir, 'models/atrlp8876_/results/squad_qa/squad_null_odds.json'),
                './data/squad_null_odds.json')
    shutil.copy(os.path.join(args.data_dir, 'models/atrlp8876_/results/squad_qa/squad_preds.json'),
                './data/squad_preds.json')

    command2 = """
    python3 ./data/run_gen_data.py   --run-type=pv --std-dev-file=%s --input-file=./data/squad_preds.json --output-file=./data/pv_dev.json
    """ % args.eval_file
    os.system(command2)

    # command3 = """
    # python3 ./data/run_gen_data.py   --run-type=reg --std-dev-file=%s --input-file=./data/dev_all_nbest.pkl --output-file=./data/reg_dev.json --split=test
    # """ % args.eval_file
    # os.system(command3)

    shutil.copy('./data/pv_dev.json', os.path.join(args.data_dir, 'finetuning_data/squad/dev.json'))

    command4 = """
    python3 run_finetuning_pv.py   --data-dir=%s --model-name=8876pv_model_   --hparams '{"model_size": "large", "task_names": ["squad"], "use_tpu": false, "eval_batch_size": 16, "predict_batch_size": 16, "max_seq_length": 512, "use_tfrecords_if_existing": false, "num_trials": 1, "do_train": false, "do_eval": true}'
    """ % args.data_dir
    os.system(command4)

    shutil.copy(os.path.join(args.data_dir, 'models/8876pv_model_/results/squad_qa/squad_null_odds.json'),
                './data/pv_squad_null_odds.json')

    # shutil.copy('./data/reg_dev.json', os.path.join(args.data_dir, 'finetuning_data/squad/dev.json'))

    # command5 = """
    # python3 run_finetuning_reg.py   --data-dir=%s --model-name=8876reg_model_   --hparams '{"model_size": "large", "task_names": ["squad"], "use_tpu": false, "eval_batch_size": 16, "predict_batch_size": 16, "max_seq_length": 512, "use_tfrecords_if_existing": false, "num_trials": 1, "do_train": false, "do_eval": true}'
    # """ % args.data_dir
    # os.system(command5)

    # shutil.copy(os.path.join(args.data_dir, 'models/8876reg_model_/results/squad_qa/f1_predict_results.pkl'),
    #             './data/dev_f1_predict_results.pkl')

    command6 = """
    python3 ./data/run_verifier.py --eval-file=%s --data-dir=./data --output-file=%s
    """ % (args.eval_file, args.output_file)
    os.system(command6)


if __name__ == '__main__':
    main()
