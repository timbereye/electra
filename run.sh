DATA_DIR="gs://squad_cx/EData"
python3 run_finetuning.py --data-dir $DATA_DIR --model-name electra_base --hparams '{"model_size": "base", "use_tpu": true, "num_tpu_cores": 8, "tpu_name": "c1","use_tfrecords_if_existing":true, "task_names": ["squad"]}'
