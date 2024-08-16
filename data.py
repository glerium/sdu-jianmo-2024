import numpy as np
from typing import Tuple, Union, Dict, List
from torch import Tensor
import torch

def get_data(year_list: Union[range, List]):
    # print(len(year_list))
    def get_data_(year):
        oridata = load_data([year])
        lst = []
        add_data = {}
        for k, v in oridata.items():
            if '0' <= k[0] <= '9':
                add_data['_'+k] = v
                lst.append(k)
        for k, v in add_data.items():
            oridata[k] = v
        del add_data
        for k in lst:
            oridata.pop(k)
        del lst
        globals().update((k, v) for k, v in oridata.items())
        n_data = land_sea_mask.shape[0]
        keys = ['land_sea_mask', 'orography', 'lattitude', 'geopotential_1000', 'relative_humidity_1000', '_2m_temperature', '_10m_u_component_of_wind', '_10m_v_component_of_wind']
        catdata = torch.empty((8760, 32, 64, 0), dtype=torch.float32)
        for key in keys:
            oridata[key] = torch.tensor(oridata[key][:, :, :, None], dtype=torch.float32)
            catdata = torch.cat((catdata, oridata[key]), dim=3)
        catdata = catdata.permute(0, 2, 1, 3)
        return catdata
    
    year_list = list(year_list)
    for i in range(0, len(year_list) - 1):
        year = year_list[i]
        catdata = get_data_(year)
        
        try:
            year2 = year_list[i+1]
            catdata = torch.cat((catdata, get_data_(year2)), dim=0)
        except:
            pass
        
        yield catdata

def load_data(year_list: Union[range, List]) -> Dict[str, np.ndarray]:
    if isinstance(year_list, range):
        year_list = list(year_list)

    ret = None
    for i in year_list:
        data = dict(np.load(f'./data/{i}.npz'))
        for key in data.keys():
            data[key] = data[key][:, 0]
        n_frame = data[next(iter(data))].shape[0]
        data['day_of_year'] = np.arange(n_frame) // 24 + 1
        data['time_of_day'] = np.arange(n_frame) % 24
        if ret is None:
            ret = data
        else:
            for key in ret.keys():
                ret[key] = np.concatenate((ret[key], data[key]), axis=0)

    return ret
