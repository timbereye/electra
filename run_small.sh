DATA_DIR="gs://squad_cx/EData"
python3 run_finetuning.py --data-dir $DATA_DIR --model-name electra_small --hparams '{"train_batch_size":32, "eval_batch_size":32, "predict_batch_size":32, "model_size": "small", "use_tpu": true, "num_tpu_cores": 8, "tpu_name": "c2","use_tfrecords_if_existing":true, "task_names": ["squad"]}'