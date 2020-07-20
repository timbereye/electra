
gs_data_root="gs://squad_cx/EData_pv/albert"
output_dir="output_xxlarge_2_384_2e-5"
features_dir="features_384"
albert_config_file="albert_config.json"
train_feature_file="train_feature_file"
predict_feature_file="predict_feature_file"
predict_feature_left_file="predict_feature_left_file"
init_checkpoint="model.ckpt-best"
spm_model_file="30k-clean.model"
albert_config_file_gs=${gs_data_root}"/albert_xxlarge/"${albert_config_file}
output_dir_gs=${gs_data_root}"/"$output_dir
features_dir_gs=${gs_data_root}"/"${features_dir}
train_file_gs="gs://squad_cx/EData_pv/finetuning_data/squad/train.json"
predict_file_gs="gs://squad_cx/EData_pv/finetuning_data/squad/dev.json"
train_feature_file_gs=${features_dir_gs}"/"${train_feature_file}
predict_feature_file_gs=${features_dir_gs}"/"${predict_feature_file}
predict_feature_left_file_gs=${features_dir_gs}"/"${predict_feature_left_file}
init_checkpoint_gs=${gs_data_root}"/albert_xxlarge/"${init_checkpoint}
spm_model_file_driver=${spm_model_file}

python3 run_squad_v2.py \
  --albert_config_file=$albert_config_file_gs \
  --output_dir=$output_dir_gs \
  --train_file=$train_file_gs \
  --predict_file=$predict_file_gs \
  --train_feature_file=$train_feature_file_gs \
  --predict_feature_file=$predict_feature_file_gs \
  --predict_feature_left_file=$predict_feature_left_file_gs \
  --init_checkpoint=$init_checkpoint_gs \
  --spm_model_file=$spm_model_file_driver \
  --do_lower_case \
  --max_seq_length=384 \
  --doc_stride=128 \
  --max_query_length=64 \
  --do_train \
  --do_predict \
  --train_batch_size=32 \
  --predict_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=2.0 \
  --warmup_proportion=.1 \
  --save_checkpoints_steps=5000 \
  --n_best_size=20 \
  --max_answer_length=30 \
  --use_tpu \
  --tpu_name "c3"