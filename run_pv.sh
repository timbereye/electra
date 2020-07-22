DATA_DIR="gs://squad_cx/EData_pv"
python3 run_finetuning.py --data-dir $DATA_DIR --model-name electra_large --hparams '{"do_train":false, "train_batch_size":32, "eval_batch_size":32, "predict_batch_size":32, "model_size": "large", "use_tpu": true, "num_tpu_cores": 8, "tpu_name": "grpc://10.9.246.234:8470","use_tfrecords_if_existing":true, "task_names": ["squad"]}'
python3 run_finetuning.py --data-dir $DATA_DIR --model-name electra_large --hparams '{"do_train":false, "max_seq_length":384, "train_batch_size":32, "eval_batch_size":32, "predict_batch_size":32, "model_size": "large", "use_tpu": true, "num_tpu_cores": 8, "tpu_name": "grpc://10.9.246.234:8470","use_tfrecords_if_existing":true, "task_names": ["squad"]}'
python3 run_finetuning.py --data-dir $DATA_DIR --model-name electra_large --hparams '{"do_train":false, "learning_rate":3e-5, "num_train_epochs":3, "train_batch_size":32, "eval_batch_size":32, "predict_batch_size":32, "model_size": "large", "use_tpu": true, "num_tpu_cores": 8, "tpu_name": "grpc://10.9.246.234:8470","use_tfrecords_if_existing":true, "task_names": ["squad"]}'
