import argparse
import torch
import os
# from main_model import CSDI_Forecasting
from model.Diffusion import Diff_Forecasting
from data.data_load import load_dataset
from utils import train,evaluate
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np

def set_cpu_num(cpu_num):
    if cpu_num <= 0: return
    os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)
set_cpu_num(5)

def set_random_seed(seed: int):
    random.seed(seed)                        
    np.random.seed(seed)                     
    torch.manual_seed(seed)                  
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)         
        torch.cuda.manual_seed_all(seed)     
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



parser = argparse.ArgumentParser(description="NPDiff")
parser.add_argument("--model",type=str, default="CSDI",
                     help="CSDI or STID")
parser.add_argument("--data_name", type=str, default="MobileBJ")
parser.add_argument('--device', default='cuda:1', 
                    help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--nsample", type=int, default=20)
parser.add_argument("--history_len", type=int, default=12)
parser.add_argument("--predict_len", type=int, default=12)
parser.add_argument("--target_dim",type=int,default=672,
                    help="the number of different grid locations")
parser.add_argument("--val_batch_size",type=int,default=24)
parser.add_argument("--diff_layers",type=int,default=4)
parser.add_argument("--Num_Comp",type=int,default=5,
                    help="0 reprent N_m, K reprent N_K")  
parser.add_argument("--Lambda",type=float,default=0.5)  
parser.add_argument("--local_dynamics",action="store_true",default=False,
                    help="whether to use local dynamics for 12-1 prediction task")
parser.add_argument("--lr",type=float,default=0.001)
parser.add_argument("--few_shot",type=int,default=0.0)
args = parser.parse_args()
print(args)
set_random_seed(args.seed)

if args.local_dynamics & (args.predict_len==1):
    foldername = f"./save/{args.model}/prior_local_Lambda{args.Lambda}_/" + args.data_name + f'_{args.history_len}_{args.predict_len}/'
else:
    foldername = f"./save/{args.model}/prior_periodic_Comp{args.Num_Comp}_Lambda{args.Lambda}_" + args.data_name + f'_{args.history_len}_{args.predict_len}/'


print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)



train_loader, valid_loader, test_loader, val_target_tensor,test_target_tensor,scaler=load_dataset(args)




model = Diff_Forecasting(args).to(args.device)

writer=SummaryWriter(log_dir=foldername)


if args.modelfolder == "":
    train(
        model,
        args,
        train_loader,
        scaler=scaler,
        valid_target=val_target_tensor,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))



writer.close()



model.target_dim = args.target_dim


model.load_state_dict(torch.load(foldername+ "/model.pth"))
print('test dataset:')
evaluate(args.model,args.history_len,model, test_loader,test_target_tensor,scaler,nsample=args.nsample)


