#!/bin/bash
python data/data_prepare.py --dataset_name "MobileBJ" --history_seq_len 12 --future_seq_len 12
python data/data_prepare.py --dataset_name "MobileNJ" --history_seq_len 12 --future_seq_len 12
python data/data_prepare.py --dataset_name "MobileSH14" --history_seq_len 12 --future_seq_len 12
python data/data_prepare.py --dataset_name "MobileSH16" --history_seq_len 12 --future_seq_len 12


python data/data_prepare.py --dataset_name "MobileBJ" --history_seq_len 12 --future_seq_len 1
python data/data_prepare.py --dataset_name "MobileNJ" --history_seq_len 12 --future_seq_len 1
python data/data_prepare.py --dataset_name "MobileSH14" --history_seq_len 12 --future_seq_len 1
python data/data_prepare.py --dataset_name "MobileSH16" --history_seq_len 12 --future_seq_len 1