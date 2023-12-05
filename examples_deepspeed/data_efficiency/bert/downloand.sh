# 
file=examples_deepspeed/data_efficiency/bert/pile_data_download_preprocess.py
nohup python $file 0 10 20 29 \
    > $file.log 2>&1 &``