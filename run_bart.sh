python bart_for_qa.py \
--data_dir='./data/' \
--model_name_or_path=facebook/bart-base \
--learning_rate=3e-5 \
--num_train_epochs=5 \
--train_batch_size=1 \
--eval_batch_size=1 \
--test_batch_size=1 \
--n_gpu 1 \
--do_train \
--do_predict
#--max_source_length 384 \
#--max_target_length 384
#--do_predict \
#--early_stopping_patience 10 \
