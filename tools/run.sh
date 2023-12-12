# gpt2 preprocess_data
# file=tools/preprocess_data.py
file=tools/check_data_size.py
nohup python $file \
    --input /mnt/nas1/dong-qichang/corpus/wiki/wikipedia/20230601/en_wikipedia.jsonl \
    --output-prefix /mnt/nas1/dong-qichang/corpus/wiki/wikipedia/20230601/my-gpt2 \
    --vocab-file /mnt/nas1/models/gpt2/vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file /mnt/nas1/models/gpt2/merges.txt \
    --workers 24 \
    --append-eod \
    > $file.log 2>&1 &