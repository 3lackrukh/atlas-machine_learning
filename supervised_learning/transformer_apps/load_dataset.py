#!/usr/bin/env python3
import tensorflow as tf
import tensorflow_datasets as tfds

pt2en_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train[:1]', as_supervised=True)
for pt, en in pt2en_train:
  print(pt.numpy().decode('utf-8'))
  print(en.numpy().decode('utf-8'))