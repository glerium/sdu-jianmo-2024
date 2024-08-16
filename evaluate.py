import torch
import torch.nn as nn
import numpy as np
from data import get_data

w = None

def rmse(v: int, x_pred: torch.Tensor, x_true: torch.Tensor):
    ret = 0xffffffff
    n_data = x_pred.shape[0]
    for i in range(0, n_data - 7, 7):
        now = 0
        for j in range(i, i+7):
            now += torch.sqrt(torch.sum((x_pred[i,:,:,v] - x_true[i,:,:,v]) ** 2 * w)) / torch.sqrt(w.sum())
        ret = min(ret, now)
    return ret

def bias(v: int, x_pred: torch.Tensor, x_true: torch.Tensor):
    ret = 0xffffffff
    n_data = x_pred.shape[0]
    for i in range(0, n_data - 7, 7):
        now = 0
        for j in range(i, i+7):
            now += torch.sum(torch.abs(x_pred[i,:,:,v] - x_true[i,:,:,v]) * w) / w.sum()
        ret = min(ret, now)
    return ret

def acc(v: int, x_pred: torch.Tensor, x_true: torch.Tensor):
    n_data = x_pred.shape[0]
    ret = 0
    for i in range(0, n_data - 7, 7):
        now = 0
        for j in range(i, i+7):
            mean_x_pred = x_pred[i,:,:,v].mean()
            mean_x_true = x_true[i,:,:,v].mean()
            xx = torch.sum(w * (x_pred[i,:,:,v] - mean_x_pred) * (x_true[i,:,:,v] - mean_x_true))
            yy = torch.sqrt(torch.sum(w * ((x_pred[i,:,:,v] - mean_x_pred) ** 2)) * torch.sum(w * ((x_true[i,:,:,v] - mean_x_true) ** 2)))
            now += xx / yy
        ret = max(ret, now)
    return ret

def evaluate(model_file: str):
    model = torch.load(model_file)
    model.cuda()
    model.eval()

    print(f'Validating {model_file}...')
    window_size = 168
    data_iter = get_data([2017, 2018])
    x_true = None
    x_pred = None
    for year, catdata in enumerate(data_iter, start=2017):
        for idx, i in enumerate(range(0, catdata.shape[0] - 168 * 2, 168)):
            if x_true is None:
                x_true = torch.empty((len(range(0, catdata.shape[0] - 168 * 2, 168)), 168, 64, 32, 3))
                x_pred = torch.empty((len(range(0, catdata.shape[0] - 168 * 2, 168)), 168, 64, 32, 3))
            input = catdata[None, i:i+window_size, :, :, :].cuda()
            input[:, :, :, :, 1] = (input[:, :, :, :, 1] + 30) / 5300
            input[:, :, :, :, 2] = input[:, :, :, :, 2] / 180 + 1
            input[:, :, :, :, 3] = (input[:, :, :, :, 3] + 8000) / 14000
            input[:, :, :, :, 4] = (input[:, :, :, :, 4] + 10) / 190
            input[:, :, :, :, -3] = (input[:, :, :, :, -3] - 190) / 140
            input[:, :, :, :, -2:] = (input[:, :, :, :, -2:] + 50) / 100

            target = catdata[None, i+window_size:i+window_size*2, :, :, :].cuda()
            with torch.no_grad():
                output = model(input, input.clone())
            
            output[:, :, :, :, 1] = output[:, :, :, :, 1] * 5300 - 30
            output[:, :, :, :, 2] = (output[:, :, :, :, 2] - 1) * 180
            output[:, :, :, :, 3] = output[:, :, :, :, 3] * 14000 + 8000
            output[:, :, :, :, 4] = output[:, :, :, :, 4] * 190 - 10
            output[:, :, :, :, -3] = output[:, :, :, :, -3] * 140 + 190
            output[:, :, :, :, -2:] = output[:, :, :, :, -2:] * 100 - 50
            
            x_true[idx] = target[0, :, :, :, -3:].cpu()
            x_pred[idx] = output[0, :, :, :, -3:].cpu()
    
    lat = catdata[0, :, :, 2]
    global w
    if w is None:
        w = torch.cos(torch.pi / 180 * lat)

    x_pred = x_pred.reshape(-1, 64, 32, 3)[::24]
    x_true = x_true.reshape(-1, 64, 32, 3)[::24]

    funcs = [rmse, bias, acc]
    results = np.zeros((3, 3))
    for idx, func in enumerate(funcs):
        for v in range(3):
            results[idx, v] = func(v, x_pred, x_true)

    return results

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    results = []
    for year in range(1979, 1991):
        model = f'epoch0_{year}_0.00.model'
        kkk = evaluate(model)
        print(kkk)
        results.append(kkk)
    results = np.array(results)

    row_titles = ['RMSE', 'Bias', 'ACC']
    col_titles = ['temperature', 'u_component_of_wind', 'v_component_of_wind']
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for j in range(3):
        for i in range(3):
            axs[j].plot(list(range(12)), results[:,j,i], label=col_titles[i])
        axs[j].legend()
        axs[j].set_title(f'{row_titles[j]}', fontweight='bold')
        axs[j].set_xlabel('Epoch')
        axs[j].set_ylabel(row_titles[j])
    axs[2].set_ylim((0, 10))
    plt.savefig('results.png')
