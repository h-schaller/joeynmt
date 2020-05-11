#! /bin/bash

toy_model_scripts=`dirname "$0"`
base=$toy_model_scripts/../..

models=$base/models
trained_models=$base/trained_models
configs=$base/configs/toy_models_configs
translations=$base/translations

mkdir -p $translations
mkdir -p $models
mkdir -p trained_models

num_threads=12
device=0

# measure time

SECONDS=0

# train factor models:
CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python -m joeynmt train $configs/rnn_wmt16_factors_concatenate_deen1.yaml
source $base/scripts/toy_model_scripts/evaluate.sh rnn_wmt16_factors_concatenate_deen1
cp -r $models/rnn_wmt16_factors_concatenate_deen1 $trained_models/
rm -r $models/rnn_wmt16_factors_concatenate_deen1

CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python -m joeynmt train $configs/rnn_wmt16_factors_concatenate_deen2.yaml
source $base/scripts/toy_model_scripts/evaluate.sh rnn_wmt16_factors_concatenate_deen2
cp -r $models/rnn_wmt16_factors_concatenate_deen2 $trained_models/
rm -r $models/rnn_wmt16_factors_concatenate_deen2

CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python -m joeynmt train $configs/rnn_wmt16_factors_concatenate_deen3.yaml
source $base/scripts/toy_model_scripts/evaluate.sh rnn_wmt16_factors_concatenate_deen3
cp -r $models/rnn_wmt16_factors_concatenate_deen3 $trained_models/
rm -r $models/rnn_wmt16_factors_concatenate_deen3

CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python -m joeynmt train $configs/rnn_wmt16_factors_add_deen1.yaml
source $base/scripts/toy_model_scripts/evaluate.sh rnn_wmt16_factors_add_deen1
cp -r $models/rnn_wmt16_factors_add_deen1 $trained_models/
rm -r $models/rnn_wmt16_factors_add_deen1

CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python -m joeynmt train $configs/rnn_wmt16_factors_add_deen2.yaml
source $base/scripts/toy_model_scripts/evaluate.sh rnn_wmt16_factors_add_deen2
cp -r $models/rnn_wmt16_factors_add_deen2 $trained_models/
rm -r $models/rnn_wmt16_factors_add_deen2

CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python -m joeynmt train $configs/rnn_wmt16_factors_add_deen3.yaml
source $base/scripts/toy_model_scripts/evaluate.sh rnn_wmt16_factors_add_deen3
cp -r $models/rnn_wmt16_factors_add_deen3 $trained_models/
rm -r $models/rnn_wmt16_factors_add_deen3

echo "time taken:"
echo "$SECONDS seconds"
