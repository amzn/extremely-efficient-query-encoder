extremely-efficient-online-query-encoding-for-dense-retrieval is an extension of the popular 
[Tevatron package](https://github.com/texttron/tevatron/) 
([commit](https://github.com/texttron/tevatron/commit/b8f33900895930f9886012580e85464a5c1f7e9a)),
adding the ability to use a small query encoder (with a large passage encoder) to have very low encoding time, while 
incurring minor impact in quality. 

## Instructions

1. Create (teacher) embeddings for all queries in the train set, using
   ```bash
   python -m tevatron.driver.encode --output_dir /tmp/encode --model_name_or_path Luyu/co-condenser-marco --fp16 \
   --per_device_eval_batch_size 128 \
   --encode_in_path ../resources/pretrain_data/train_queries_tokens.jsonl \
   --encoded_save_path ../resources/pretrain_data/train_queries.pt`
   ```
2. Run pretraining using `python -m run_pretraining.pretrain`
3. Run training using `marco_train_pretrained_model.sh`
4. Evaluate using `full_eval.sh`

## Citations

```
@article{cohen2024indi,
  title={Extremely efficient online query encoding for dense retrieval},
  author={Cohen, Nachshon and Fairstein, Yaron and Kushilevitz, Guy},
  booktitle={Findings of the Association for Computational Linguistics: NAACL 2024},
  year={2024}
}
```
