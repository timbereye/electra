DATA_DIR="gs://squad_cx/EData"
python3 run_finetuning.py --data-dir $DATA_DIR --model-name electra_small --hparams '{"train_batch_size":8, "eval_batch_size":8, "model_size": "small", "use_tpu": false, "num_tpu_cores": 8, "tpu_name": "c1","use_tfrecords_if_existing":true, "task_names": ["squad"]}'
