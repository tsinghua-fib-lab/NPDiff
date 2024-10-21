from model.CSDI import CSDI
from model.STID import STID


class Denoising_network():
    def __init__(self) -> None:
        self.model_dict = {
            "CSDI": CSDI,
            "STID": STID,
        }
    def get_model(self,args):
        if args.model not in self.model_dict:
            raise ValueError(f"Model {args.model} not found")
        if args.model == "CSDI":
            return self.model_dict[args.model](args,inputdim=2)      
        elif args.model == "STID":
            return self.model_dict[args.model](args)