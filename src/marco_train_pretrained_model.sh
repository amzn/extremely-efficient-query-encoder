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

HN_FOLDER="../examples/coCondenser-marco/marco/bert/train-hn"
OUT_FOLDER="../outputs/models/reproduce"

if [ ! -e $HN_FOLDER ] || [ ! -e $OUT_FOLDER ]; then
  echo "please ensure hard negative folder and output folder are set appropriately"
  echo "hn folder is created by following examples/coCondenser-marco/README.md"
  echo "output folder is where you want the train results to reside. Simply create it."
  exit 1
fi


EPOCHS=3
MODEL_LOC=$1  # get this from running run_pretrain.pretrain
TP_MODEL="Luyu/co-condenser-marco"


MODEL_NAME="${MODEL_LOC##*/}"
PRETRAIN_PATH_PARAM="--rnn_pretrained_path ${MODEL_LOC}"
TRAINING_MODEL_NAME="${MODEL_NAME}_e${EPOCHS}_${TP}"

if [ -z "$MODEL_NAME" ]; then
  echo "put pretrained model path without a trailing /"
  exit 1
fi

echo "Running ${MODEL_NAME} with ${EPOCHS} epochs."

# Training regular model.
python -m tevatron.driver.train --output_dir "${OUT_FOLDER}/${TRAINING_MODEL_NAME}" \
--model_name_or_path ${TP_MODEL} --freeze_passage_enc false --save_steps 62867 \
--train_dir ${HN_FOLDER} \
--fp16 --per_device_train_batch_size 8 --learning_rate 5e-6 --num_train_epochs ${EPOCHS} --dataloader_num_workers 3 \
--rnn_query true ${PRETRAIN_PATH_PARAM}

./full_eval.sh ${OUT_FOLDER}/"${TRAINING_MODEL_NAME}"
