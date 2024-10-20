import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
from einops import rearrange



class EarlyStopping():
    def __init__(self, patience=7, delta=0, path='model.pth', verbose=False):

        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss







def train(
    model,
    args,
    train_loader,
    scaler=None,
    valid_target=None,
    valid_loader=None,
    valid_epoch_interval=20,
    foldername="",
):
    
    early_stopping = EarlyStopping(patience=5, verbose=True,path=(foldername+'/model.pth'))

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)


    if foldername != "":
        output_path = foldername + "/model.pth"




    p1 = int(0.4 * 100)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1], gamma=0.4
    )


    best_valid_loss = 1e10
    for epoch_no in range(100): #config["epochs"]
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )

            lr_scheduler.step()

        mae_val=evaluate(args.model,args.history_len,model, valid_loader,valid_target,scaler,nsample=3)
        early_stopping(mae_val, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):

    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):

    eval_points = eval_points.mean(-1)
    target = target * scaler + mean_scaler
    target = target.sum(-1)
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1),quantiles[i],dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)



def mae(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred)).item()

def rmse(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()

def smape(y_true, y_pred):
    y_true = y_true.float()
    y_pred = y_pred.float()
    smape_value = torch.mean(2.0 * torch.abs(y_true - y_pred) / (torch.abs(y_true) + torch.abs(y_pred)))    
    return smape_value.item()

def metric_our(pred,real):
    mae1 = mae(pred, real)
    rmse1 = rmse(pred, real)
    smape1= smape(pred, real)
    return mae1,smape1,rmse1




def evaluate(model_name,history_len,model, test_loader,target,scaler,nsample=10):


    mae_loss=[]
    rmse_loss=[]
    with torch.no_grad():
        model.eval()
        rmse_total = 0
        mae_total = 0
        smape_total=0
        predict=[]

        if (model_name=='STID') or (model_name=='CSDI'):
            target=rearrange(target,'b l h w -> b (h w) l')
            with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
                for batch_no, test_batch in enumerate(it, start=1):
                    output = model.evaluate(test_batch, nsample) # B, n_samples, K, L
                    predict.append(output[:,:,:,history_len:])

                predict=torch.cat(predict,dim=0) # N, n_samples, K, L
                predict=scaler.inverse_transform(predict)
            for n in range(nsample):
                pre_data=predict[:,n,:,:]
                rmse_total+=rmse(pre_data,target)
                mae_total+=mae(pre_data,target)
                smape_total+=smape(pre_data,target)
                rmse_loss.append(rmse(pre_data,target))
                mae_loss.append(mae(pre_data,target))
        elif model_name=='ConvLSTM':
            target=rearrange(target,'b l h w -> b h w l')
            with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
                for batch_no, test_batch in enumerate(it, start=1):
                    output = model.evaluate(test_batch, nsample) # B, n_samples, K, L
                    predict.append(output[:,:,:,:,history_len:])

                predict=torch.cat(predict,dim=0) # N, n_samples, K, L
                predict=scaler.inverse_transform(predict) # N, n_samples, H,W, L
            for n in range(nsample):
                pre_data=predict[:,n,:,:,:]
                rmse_total+=rmse(pre_data,target)
                mae_total+=mae(pre_data,target)
                smape_total+=smape(pre_data,target)
                rmse_loss.append(rmse(pre_data,target))
                mae_loss.append(mae(pre_data,target))
        else:
            print('model name error')

    print('metric:')
    print(f'mae: {mae_total/nsample:.4f} rmse: {rmse_total/nsample:.4f}')
    # print(f'mae_loss: {mae_loss} \n rmse_loss: {rmse_loss}')
    return mae_total/nsample

