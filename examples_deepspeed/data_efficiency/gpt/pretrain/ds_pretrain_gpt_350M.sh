# CL+random-LTD 150B tokens (50%):
lr=4.0e-4
train_tokens_in_billion=150
ltd_enabled="true"
ltd_start=128
ltd_step=100000
cl_enabled="true"
cl_num_metric=2
cl_1st_metric="voc"
cl_1st_index_to_sample_path="/mnt/nas1/dong-qichang/corpus/wiki/wikipedia/20230601/analysis_pile_gpt_1epoch/vocab_rarity/vocab_rarity_index_to_sample_percentile_merged"
cl_1st_index_to_metric_path="/mnt/nas1/dong-qichang/corpus/wiki/wikipedia/20230601/analysis_pile_gpt_1epoch/vocab_rarity/vocab_rarity_index_to_metric"
cl_1st_difficulty_type="percentile"
cl_1st_clustering_type="schedule_based"
cl_1st_min=1
cl_1st_max=100
cl_1st_total_step=55000
cl_1st_difficulty_step=1
cl_1st_root=2
cl_2nd_metric="seqlen_truncate"
cl_2nd_index_to_sample_path="dummy"
cl_2nd_index_to_metric_path="dummy"
cl_2nd_difficulty_type="value"
cl_2nd_clustering_type="single_cluster"
cl_2nd_min=80
cl_2nd_max=2048
cl_2nd_total_step=55000
cl_2nd_difficulty_step=8
cl_2nd_root=1
bash ds_pretrain_gpt_1.3B_dense_base_script.sh ${lr} \
    ${train_tokens_in_billion} ${ltd_enabled} ${ltd_start} ${ltd_step} \
    ${cl_enabled} ${cl_num_metric} ${cl_1st_metric} \
    ${cl_1st_index_to_sample_path} ${cl_1st_index_to_metric_path} \
    ${cl_1st_difficulty_type} ${cl_1st_clustering_type} ${cl_1st_min} \
    ${cl_1st_max} ${cl_1st_total_step} ${cl_1st_difficulty_step} \
    ${cl_1st_root} ${cl_2nd_metric} ${cl_2nd_index_to_sample_path} \
    ${cl_2nd_index_to_metric_path} ${cl_2nd_difficulty_type} \
    ${cl_2nd_clustering_type} ${cl_2nd_min} ${cl_2nd_max} \
    ${cl_2nd_total_step} ${cl_2nd_difficulty_step} ${cl_2nd_root}