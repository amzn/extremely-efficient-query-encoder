#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
MODEL=$1
OUT_FOLD=/tmp/"$RANDOM"
mkdir -p "$OUT_FOLD"
echo $MODEL > "$OUT_FOLD"/model.txt
echo "Encoding queries-----------"
python -m tevatron.driver.encode --output_dir /tmp/encode --model_name_or_path $MODEL --fp16 --q_max_len 32 \
      --encode_is_qry --per_device_eval_batch_size 128 --encode_in_path ../resources/dev7k.query.json \
      --encoded_save_path "$OUT_FOLD"/encoded_queries.pt
if [ $? -ne 0 ]; then echo "error encoding queries"; rm -r "$OUT_FOLD"; exit 1; fi
echo "Finished encoding queries, encoding passages-----------"
for i in $(seq -f "%02g" 0 9)
do
  python -m tevatron.driver.encode --output_dir /tmp/encode --model_name_or_path $MODEL --fp16 \
        --per_device_eval_batch_size 128 \
        --encode_in_path ../examples/coCondenser-marco/marco/bert/corpus/split${i}.json \
        --encoded_save_path "$OUT_FOLD"/split${i}.pt
done
if [ $? -ne 0 ]; then echo "error encoding passages"; rm -r "$OUT_FOLD"; exit 1; fi
echo "Finished encoding passages-----------"
python -m tevatron.faiss_retriever --query_reps "$OUT_FOLD"/encoded_queries.pt --passage_reps "$OUT_FOLD"/'split*.pt' \
      --depth 1000 --batch_size -1 --save_text --save_ranking_to "$OUT_FOLD"/dev.rank.tsv
if [ $? -ne 0 ]; then echo "error faiss retrieve"; rm -r "$OUT_FOLD"; exit 1; fi

python ../examples/coCondenser-marco/score_to_marco "$OUT_FOLD"/dev.rank.tsv