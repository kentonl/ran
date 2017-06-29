## Introduction

This codebase supports replication of the language modeling results in [Recurrent Additive Networks](https://arxiv.org/abs/1705.07393) ([Kenton Lee](http://www.kentonl.com), [Omer Levy](https://levyomer.wordpress.com), and [Luke Zettlemoyer](https://www.cs.washington.edu/people/faculty/lsz)).

## Recurrent Additive Networks
The TensorFlow implementation of Recurrent Additive Networks (RAN) is found in `ran.py` and is used by the experiments in the subdirectories.

## Experiments
### Penn Treebank:
The word-level language modeling for Penn Treebank is found under the `ptb` directory. This code is derived from https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb.

#### Data preparation
* ```curl -O http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz```
* ```mkdir data```
* ```tar -xzvf simple-examples.tgz -C data```

#### Train and Evaluate
* `python -m ptb.ptb_word_lm --data_path=data/simple-examples/data --model=tanh_medium`

Replace `tanh_medium` with the desired setting.

### Billion-word Benchmark:
The word-level language modeling for the billion-word benchmark is found under the `bwb` directory. This code is derived from https://github.com/rafaljozefowicz/lm.

#### Data preparation
* ```curl -O http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz```
* ```mkdir data```
* ```tar -xzvf 1-billion-word-language-modeling-benchmark-r13output.tar.gz -C data```
* ```curl -o data/1-billion-word-language-modeling-benchmark-r13output/1b_word_vocab.txt https://raw.githubusercontent.com/rafaljozefowicz/lm/master/1b_word_vocab.txt```

#### Train
* `CUDA_VISIBLE_DEVICES=0,1 python -m bwb.single_lm_train --logdir logs --num_gpus 2 --hpconfig num_shards=2 --datadir data/1-billion-word-language-modeling-benchmark-r13output`
#### Evaluate
* `CUDA_VISIBLE_DEVICES= python -m bwb.single_lm_train --logdir logs --mode eval_test_ave --hpconfig num_shards=2 --datadir data/1-billion-word-language-modeling-benchmark-r13output`

### Text8:
The character-level language modeling for Text8 is found under the `text` directory. This code is derived from https://github.com/julian121266/RecurrentHighwayNetworks

#### Data Preparation
* ```curl -O http://mattmahoney.net/dc/text8.zip```
* ```mkdir data```
* ```unzip text8.zip -d data```

#### Train and Evaluate
* `python -m text8.char_train`
