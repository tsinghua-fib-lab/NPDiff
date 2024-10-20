from model.ConvLSTM import ConvLSTM
from model.CSDI import CSDI
from model.STID import STID


class Denoising_network():
    def __init__(self) -> None:
        self.model_dict = {
            "ConvLSTM": ConvLSTM,
            "CSDI": CSDI,
            "STID": STID,
        }
    def get_model(self,args):
        if args.model not in self.model_dict:
            raise ValueError(f"Model {args.model} not found")
        if args.model == "ConvLSTM":
            return self.model_dict[args.model](args,input_dim=64, hidden_dim=64, kernel_size=(3, 3), num_layers=3,
                 batch_first=True, bias=True, return_all_layers=False)
        elif args.model == "CSDI":
            return self.model_dict[args.model](args,inputdim=2)
        elif args.model == "STID":
            return self.model_dict[args.model](args)