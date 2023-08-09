**continual_pretraining.ipynb**: Code to continually pretrain using salient span masking  
**query_metaqamultiT2.py**: Code for Question Booster  
**run_seq2seq_cbqa.py**: To finetune on closed book QA i.e. without context   
**run_seq2seq_qa.py**: To finetune with context  
**trainer_seq2seq_cbqa.py**: Allows to change functions of trainer for run_seq2seq_cbqa  
**trainer_seq2seq_qa.py**: Allows to change functions of trainer for run_seq2seq_cbqa  
Fine tune code is from: https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering and instructions to run are available there.  
Example Commands:  
CUDA_VISIBLE_DEVICES=3,5 python run_seq2seq_cbqa.py   --model_name_or_path ../SKILL/aviation/trained_models/T5_large_with_MetaTriples/20  --train_file data/metaqa/3hop/qa_train_triples.csv --validation_file data/metaqa/3hop/qa_dev_triples.csv  --test_file  data/metaqa/3hop/qa_test_triples.csv  --preprocessing_num_workers 10     --question_column question   --answer_column answer   --do_train     --per_device_train_batch_size 128 --per_device_eval_batch_size 32 --num_train_epochs 20   --max_seq_length 128   --doc_stride 128 --output_dir models/metaqa_3hop_kg_unverb_20/ --save_steps 1000  --seed 42 --overwrite_output_dir --report_to wandb --logging_steps 100  --learning_rate 1e-3 --tokenizer_name t5-large   

CUDA_VISIBLE_DEVICES=0 python run_seq2seq_qa.py   --model_name_or_path t5-large --train_file data/metaQA/2hop/qa_train_triples_multitop3.csv --validation_file data/metaQA/2hop/qa_dev_triples_multitop3.csv  --test_file  data/metaQA/2hop/qa_test_triples_multitop3.csv --overwrite_cache --preprocessing_num_workers 10     --question_column question    --context_column context  --answer_column answer   --do_train     --per_device_train_batch_size 128 --per_device_eval_batch_size 32 --num_train_epochs 20   --max_seq_length 128   --doc_stride 128 --output_dir models/metaqa_2hop_top3_COLBERT_multihop/ --save_steps 200  --seed 42 --overwrite_output_dir --report_to wandb --logging_steps 100  --learning_rate 1e-3 --tokenizer_name t5-large  
