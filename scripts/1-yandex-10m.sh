#!/usr/bin/env bash

python main.py -m \
  experiment=1-yandex-10m/1/ \
  model=GCTR,DCTR,RCTR,CM,PBM,UBM,DBN,SDBN,DCM,CCM \
  dataset=/ivi/ilps/datasets/yandex/relevance_prediction/YandexClicks.txt \
  train_sessions=[0,10_000_000] \
  val_sessions=[10_000_000,15_000_000] \
  test_sessions=[15_000_000,20_000_000] \
  min_train_sessions_per_eval_query=10 \
  $@

python main.py -m \
  experiment=1-yandex-10m/2/ \
  model=GCTR,DCTR,RCTR,CM,PBM,UBM,DBN,SDBN,DCM,CCM \
  dataset=/ivi/ilps/datasets/yandex/relevance_prediction/YandexClicks.txt \
  train_sessions=[10_000_000,20_000_000] \
  val_sessions=[20_000_000,25_000_000] \
  test_sessions=[25_000_000,30_000_000] \
  min_train_sessions_per_eval_query=10 \
  $@

python main.py -m \
  experiment=1-yandex-10m/3/ \
  model=GCTR,DCTR,RCTR,CM,PBM,UBM,DBN,SDBN,DCM,CCM \
  dataset=/ivi/ilps/datasets/yandex/relevance_prediction/YandexClicks.txt \
  train_sessions=[20_000_000,30_000_000] \
  val_sessions=[30_000_000,35_000_000] \
  test_sessions=[35_000_000,40_000_000] \
  min_train_sessions_per_eval_query=10 \
  $@
