import torch
import torch.nn as nn
import numpy as np
from data import get_data
from model import Model

model = Model().cuda()
# model = torch.load('epoch0_1993_0.00.model')
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-4)

print('Start training...')
for epoch in range(10):
    data_iter = get_data(range(1979, 2017))
    print(f'\nEpoch {epoch}:')
    overall_loss = 0
    overall_cnt = 0
    for year, catdata in enumerate(data_iter, start=1979):
        print(f'Year {year}')
        year_loss = 0
        year_cnt = 0
        for i in range(catdata.shape[0] - 168 * 2):
            if year_cnt != 0 and i % 100 == 0:
                print(f"step={i}, loss={year_loss/year_cnt:.4f}")
            optimizer.zero_grad()
            input = catdata[None, i:i+168, :, :, :].cuda()
            input[:, :, :, :, 1] = (input[:, :, :, :, 1] + 30) / 5300
            input[:, :, :, :, 2] = input[:, :, :, :, 2] / 180 + 1
            input[:, :, :, :, 3] = (input[:, :, :, :, 3] + 8000) / 14000
            input[:, :, :, :, 4] = (input[:, :, :, :, 4] + 10) / 190
            input[:, :, :, :, -3] = (input[:, :, :, :, -3] - 190) / 140
            input[:, :, :, :, -2:] = (input[:, :, :, :, -2:] + 50) / 100
            
            target = catdata[None, i+168:i+168*2, :, :, :].cuda()
            target[:, :, :, :, 1] = (target[:, :, :, :, 1] + 30) / 5300
            target[:, :, :, :, 2] = target[:, :, :, :, 2] / 180 + 1
            target[:, :, :, :, 3] = (target[:, :, :, :, 3] + 8000) / 14000
            target[:, :, :, :, 4] = (target[:, :, :, :, 4] + 10) / 190
            target[:, :, :, :, -3] = (target[:, :, :, :, -3] - 190) / 140
            target[:, :, :, :, -2:] = (target[:, :, :, :, -2:] + 50) / 100
            
            output = model(input, input.clone())
            
            output = output[:, :, :, :, -5:]
            target = target[:, :, :, :, -5:]
            loss = loss_fn(output, target)
            
            loss.backward()
            optimizer.step()
            year_loss += loss.item()
            year_cnt += 1
            overall_loss += loss.item()
            overall_cnt += 1
        if year_cnt == 0:
            continue
        print(f'Year {year}, Loss={year_loss / year_cnt}')
        torch.save(model, f'epoch{epoch}_{year}.model')
    print(f'Epoch {epoch}, Loss={overall_loss / overall_cnt}')

print('Finished training.')
