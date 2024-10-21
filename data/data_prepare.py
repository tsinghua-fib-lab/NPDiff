import os
import sys
import pickle
import argparse
import numpy as np
sys.path.append(os.path.abspath(__file__ + "/../../../.."))
from transform import standard_transform


def generate_data(args: argparse.Namespace):
    """Preprocess and generate train/valid/test datasets.
        - Normalization method: standard norm.
        - Dataset division: 6:2:2.
        - Window size: history 12, future 12.
        - Channels (features): three channels [traffic speed, day of week, time of day]
        - Target: predict the traffic speed of the future 12 time steps.

    Args:
        args (argparse): configurations of preprocessing
    """

    future_seq_len = args.future_seq_len
    history_seq_len = args.history_seq_len
    add_time_of_day = args.tod
    add_day_of_week = args.dow
    output_dir = args.output_dir
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    data_file_path = args.data_file_path
    ts_file_path=args.ts_file_path

    # read data
    data = np.load(data_file_path) # T,K,C

    ts=np.load(ts_file_path) # T,2
    time_step=ts
    print("time stamp shape: {0}".format(time_step.shape))
    print("raw time series shape: {0}".format(data.shape))

    l, n, f = data.shape
    num_samples = l - (history_seq_len + future_seq_len) + 1

    train_num_short = round(num_samples * train_ratio)
    valid_num_short = round(num_samples * valid_ratio)
    test_num_short = num_samples - train_num_short - valid_num_short

    print("number of training samples:{0}".format(train_num_short))
    print("number of validation samples:{0}".format(valid_num_short))
    print("number of test samples:{0}".format(test_num_short))


    index_list = []
    for t in range(history_seq_len, num_samples + history_seq_len):
        index = (t-history_seq_len, t, t+future_seq_len)
        index_list.append(index)

    train_index = index_list[:train_num_short]
    valid_index = index_list[train_num_short: train_num_short + valid_num_short]
    test_index = index_list[train_num_short +
                            valid_num_short: train_num_short + valid_num_short + test_num_short]

    scaler = standard_transform
    data_norm = scaler(data, output_dir, train_index, history_seq_len, future_seq_len)

    # add external feature
    feature_list = [data_norm]
    processed_data = np.concatenate(feature_list, axis=-1)
    # dump data
    index = {}
    index["train"] = train_index
    index["valid"] = valid_index
    index["test"] = test_index
    with open(output_dir + "/index_in{0}_out{1}.pkl".format(history_seq_len, future_seq_len), "wb") as f:
        pickle.dump(index, f)

    data = {}
    data["processed_data"] = processed_data
    data['data']=data
    data['ts']=time_step
    # data['data_period']=data_period
    with open(output_dir + "/data_in{0}_out{1}.pkl".format(history_seq_len, future_seq_len), "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    # sliding window size for generating history sequence and target sequence
    HISTORY_SEQ_LEN = 12
    FUTURE_SEQ_LEN = 1

    TRAIN_RATIO = 0.6
    VALID_RATIO = 0.2

    DATASET_NAME = "MobileBJ"
    TOD = True                  # if add time_of_day feature
    DOW = True                  # if add day_of_week feature
    OUTPUT_DIR = f"data/dataset"
    DATA_FILE_PATH = f"data/mobile_npy"
    TS_FILE_PATH= f"data/mobile_npy"

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str,
                        default=DATASET_NAME, help="Dataset name.")
    parser.add_argument("--output_dir", type=str,
                        default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--data_file_path", type=str,
                        default=DATA_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--history_seq_len", type=int,
                        default=HISTORY_SEQ_LEN, help="Sequence Length.")
    parser.add_argument("--future_seq_len", type=int,
                        default=FUTURE_SEQ_LEN, help="Sequence Length.")
    parser.add_argument("--tod", type=bool, default=TOD,
                        help="Add feature time_of_day.")
    parser.add_argument("--dow", type=bool, default=DOW,
                        help="Add feature day_of_week.")
    parser.add_argument("--ts_file_path", type=str,
                        default=TS_FILE_PATH, help="ts_file_path")
    parser.add_argument("--train_ratio", type=float,
                        default=TRAIN_RATIO, help="Train ratio")
    parser.add_argument("--valid_ratio", type=float,
                        default=VALID_RATIO, help="Validate ratio.")
    args = parser.parse_args()

    args.output_dir=args.output_dir+f"/{args.dataset_name}_{args.history_seq_len}_{args.future_seq_len}/"
    args.data_file_path=args.data_file_path+f"/{args.dataset_name}.npy"
    args.ts_file_path=args.ts_file_path+f"/{args.dataset_name}_ts.npy"

    
    # print args
    print("-"*(20+45+5))
    for key, value in sorted(vars(args).items()):
        print("|{0:>20} = {1:<45}|".format(key, str(value)))
    print("-"*(20+45+5))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    generate_data(args)
