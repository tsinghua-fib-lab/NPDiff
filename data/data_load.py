
import torch
from torch.utils.data import Dataset,DataLoader
import pickle
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import os
def load_pkl(pickle_file: str) -> object:
    """Load pickle data.
    Args:
        pickle_file (str): file path
    Returns:
        object: loaded objected
    """
    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print("Unable to load data ", pickle_file, ":", e)
        raise
    return pickle_data

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    

class STDataset(Dataset):
    """Time series forecasting dataset."""

    def __init__(self, NUM : int,data_name: str, history_len: int,predict_len:int, mode: str,few_shot=0.0,DEVICE='cuda:0') -> None:
        super().__init__()

        data_file_path=os.path.join('data/dataset',data_name+f'_{history_len}_{predict_len}',f"data_in{history_len}_out{predict_len}.pkl")
        index_file_path=os.path.join('data/dataset',data_name+f'_{history_len}_{predict_len}',f"index_in{history_len}_out{predict_len}.pkl")
        scaler_file_path=os.path.join('data/dataset',data_name+f'_{history_len}_{predict_len}',f"scaler_in{history_len}_out{predict_len}.pkl")

        assert mode in ["train", "val", "test"], "error mode"
        self._check_if_file_exists(data_file_path, index_file_path,scaler_file_path)
        # read raw data (normalized)
        data = load_pkl(data_file_path)
        processed_data = data['processed_data'] # n,h,w
        
        L,H,W=processed_data.shape
        num_samples = L - (history_len + predict_len) + 1
        train_num_short = round(num_samples * 0.6)
        data_train=processed_data[:train_num_short] # l,h,w

        Points = 24 if data_name != "MobileSH16" else 96

        prior_results = []
        if (data_train.shape[0] // (7 * Points))>0:
            L_series=7*Points
            for j in range(data_train.shape[0] // (7 * Points)):
                data_train_i=data_train[j*7*Points:(j+1)*7*Points,:,:].reshape(7*Points,-1)

                for i, series in enumerate(data_train_i.transpose(1,0)):
                    fft_result = np.fft.fft(series)
                    magnitudes = np.abs(fft_result)
                    mean_magnitude = np.mean(magnitudes)
                    if NUM==0:
                        idx = np.where(magnitudes > mean_magnitude)[0]
                    else:
                        idx = np.argsort(magnitudes)[-NUM:]
                    prior_fft = np.zeros_like(fft_result)
                    prior_fft[idx] = fft_result[idx]
                    prior = np.fft.ifft(prior_fft).real
                    prior_results.append(prior)
        else:
            L_series=Points
            for j in range(data_train.shape[0] // (Points)):
                data_train_i=data_train[j*Points:(j+1)*Points,:,:].reshape(Points,-1)
                for i, series in enumerate(data_train_i.transpose(1,0)):
                    fft_result = np.fft.fft(series)
                    magnitudes = np.abs(fft_result)
                    mean_magnitude = np.mean(magnitudes)
                    if NUM==0:
                        idx = np.where(magnitudes > mean_magnitude)[0]
                    else:
                        idx = np.argsort(magnitudes)[-NUM:]
                    prior_fft = np.zeros_like(fft_result)
                    prior_fft[idx] = fft_result[idx]
                    prior = np.fft.ifft(prior_fft).real
                    prior_results.append(prior)

        prior = np.mean(np.array(prior_results).reshape(-1,H*W,L_series), axis=0) # H*W,L
        prior=prior.reshape(H,W,L_series).transpose(2,0,1) # L,H,W
        prior= np.tile(prior, (int(np.ceil(L / prior.shape[0])), 1, 1))[:L, :, :] # L,H,W

        self.prior=torch.from_numpy(prior).float().to(DEVICE)
        assert prior.shape==processed_data.shape ,f"Shape mismatch: prior shape: {prior.shape}, processed_data shape: {processed_data.shape}"
        self.data = torch.from_numpy(processed_data).float().to(DEVICE)
        ts = data['ts'] # n
        self.ts= torch.from_numpy(ts).float().to(DEVICE)
        target_mask=torch.ones(predict_len+history_len,H,W) # l,h,w
        target_mask[-predict_len:,:,:]=0
        self.target_mask=target_mask.float().to(DEVICE)
        # read index
        if mode == 'val':
            mode = 'valid'
        self.index = load_pkl(index_file_path)[mode]
        self.history_len = history_len
        self.predict_len=predict_len
        self.std=load_pkl(scaler_file_path)['args']["std"]
        self.mean=load_pkl(scaler_file_path)['args']["mean"]
        self.scaler_ = StandardScaler(mean=self.mean, std=self.std)

        if mode == "train":
            assert (few_shot>=0)&(few_shot<=1), "few_shot should be in [0,1]"
            if few_shot > 0.0:
                self.index = self.index[:int(len(self.index) * few_shot)]
            elif few_shot == 0:
                self.index = load_pkl(index_file_path)[mode]


    def _check_if_file_exists(self, data_file_path: str, index_file_path: str,scaler_file_path:str):
        """Check if data file and index file exist.

        Args:
            data_file_path (str): data file path
            index_file_path (str): index file path

        Raises:
            FileNotFoundError: no data file
            FileNotFoundError: no index file
        """

        if not os.path.isfile(data_file_path):
            raise FileNotFoundError("Can not find data file {0}".format(data_file_path))
        if not os.path.isfile(index_file_path):
            raise FileNotFoundError("Can not find index file {0}".format(index_file_path))
        if not os.path.isfile(scaler_file_path):
            raise FileNotFoundError("Can not find scaler file {0}".format(scaler_file_path))
    
    @property
    def scaler(self):  
        return self.scaler_

    def __getitem__(self, index: int):
        """Get a sample.

        Args:
            index (int): the iteration index (not the self.index)

        Returns:
            tuple: (future_data, history_data), where the shape of each is L x N x C.
        """

        idx = list(self.index[index])
        if isinstance(idx[0], int):
            # continuous index
            # history_data = self.data[idx[0]:idx[1]]
            # future_data = self.data[idx[1]:idx[2]]
            data=self.data[idx[0]:idx[2]]
            ts=self.ts[idx[0]:idx[2]]

            prior_data=self.prior[idx[0]:idx[2]]
        
        else:
            # discontinuous index or custom index
            # NOTE: current time $t$ should not included in the index[0]
            history_index = idx[0]    # list
            assert idx[1] not in history_index, "current time t should not included in the idx[0]"
            
            history_index.append(idx[1])
            history_data = self.data[history_index]
            future_data = self.data[idx[1], idx[2]]
            
            ts=torch.cat((self.ts[history_index], self.ts[idx[1], idx[2]]), dim=0)
            data=torch.cat((history_data, future_data), dim=0)
            prior_data=torch.cat((self.prior[history_index], self.prior[idx[1], idx[2]]), dim=0)

        # mask l,h,w
        target_mask=self.target_mask
        # data l,h,w
        # ts   l,2

        s = {
            'observed_data': data, # K,L  ->  [24, 32, 32]
            'gt_mask': target_mask,# K,L  ->  [24, 32, 32]
            'timepoints': ts,      # L    ->  [24,2]
            'prior':prior_data     # L,H,W ->  [24, 32, 32]
        }
        return  s


    def __len__(self):
        """Dataset length

        Returns:
            int: dataset length
        """

        return len(self.index)


def load_dataset(args,shuffle=True):

    
    dataset = STDataset(args.Num_Comp,args.data_name, args.history_len,args.predict_len, 'train', args.few_shot,DEVICE=args.device)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)

    scaler=dataset.scaler

    dataset = STDataset(args.Num_Comp,args.data_name, args.history_len,args.predict_len, 'val', args.few_shot,DEVICE=args.device)
    val_loader = DataLoader(dataset, batch_size=args.val_batch_size, shuffle=False)

    dataset = STDataset(args.Num_Comp,args.data_name, args.history_len,args.predict_len, 'test', args.few_shot,DEVICE=args.device)
    test_loader = DataLoader(dataset, batch_size=args.val_batch_size, shuffle=False)
    all_targets=[]
    for x in test_loader:
        all_targets.append(x['observed_data'][:,args.history_len:,:,:])
    test_target_tensor = torch.cat(all_targets)
    test_target_tensor=scaler.inverse_transform(test_target_tensor)
    print('test_target_tensor size:',test_target_tensor.size())
    all_targets=[]
    for x in val_loader:
        all_targets.append(x['observed_data'][:,args.history_len:,:,:])
    val_target_tensor = torch.cat(all_targets)
    val_target_tensor=scaler.inverse_transform(val_target_tensor)
    print('val_target_tensor size:',val_target_tensor.size())    
    
    return train_loader,val_loader,test_loader,val_target_tensor,test_target_tensor,scaler


import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="CSDI")
    parser.add_argument("--config", type=str, default="csdi.yaml")
    parser.add_argument("--data_name", type=str, default="FlowNJ")
    parser.add_argument('--device', default='cuda:2', help='Device for Attack')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--unconditional", action="store_true")
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--nsample", type=int, default=100)
    parser.add_argument("--few_shot", type=float, default=0.0)
    parser.add_argument("--history_len", type=int, default=24)
    parser.add_argument("--predict_len", type=int, default=24)
    parser.add_argument("--target_dim",type=int,default=560)
    parser.add_argument("--val_batch_size",type=int,default=24)
    parser.add_argument("--diff_layers",type=int,default=4)
    parser.add_argument("--Num_Comp",type=int,default=3)  
    args = parser.parse_args()
    train_loader,val_loader,test_loader,val_target_tensor,test_target_tensor,_=load_dataset(args)
    print('')
