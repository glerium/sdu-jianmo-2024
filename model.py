import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.utils
from torch import Tensor
from typing import Tuple

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    # Patch Embedding
    self._input_layer = PatchEmbedding((2, 4, 21), 192)

    self.layer1 = nn.Transformer(d_model=256, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=192, batch_first=True)

    # Upsample and downsample
    self.downsample = DownSample(2048, 256)
    self.upsample = UpSample(256, 1792)

    # Patch Recovery
    self._output_layer = PatchRecovery((2, 4, 21), 192)


  def forward(self, input_surface: torch.Tensor, target_surface: torch.Tensor):
    '''Backbone architecture'''
    if input_surface.shape != target_surface.shape:
      raise ValueError('shape of input_surface and target_surface must be equal!')

    # Embed the input fields into patches
    x = self._input_layer(input_surface)
    y = self._input_layer(target_surface)

    x = self.downsample(x)
    y = self.downsample(y)


    x = self.layer1.encoder(x, mask=self.layer1.generate_square_subsequent_mask(x.shape[1], device='cuda')) 
    y = self.layer1.decoder(y, x, tgt_mask=self.layer1.generate_square_subsequent_mask(y.shape[1], device='cuda'))

    skip_y = y
    y = self.upsample(y)
    y = torch.concat((skip_y, y), dim=2)
    output = self._output_layer(y)
    
    return output


class PatchEmbedding(nn.Module):
  def __init__(self, patch_size: Tuple[int, int, int], dim):
    '''Patch embedding operation'''
    super().__init__()
    self.patch_size = patch_size
    self.conv_surface = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(4, 2), stride=(4, 2))
    
  def forward(self, input_surface):
    batch_size = input_surface.shape[0]
    bsize, length, _, _, _ = input_surface.shape
    input_surface = input_surface.reshape(-1, 64, 32, 8)
    input_surface = torch.permute(input_surface, (0, 3, 1, 2))
    input_surface = self.conv_surface(input_surface)
    input_surface = torch.permute(input_surface, (0, 2, 3, 1))
    input_surface = input_surface.reshape(bsize, length, -1)
    return input_surface

class DownSample(nn.Module):
  def __init__(self, in_dim, out_dim):
    super().__init__()
    self.weight = nn.Parameter(torch.randn((in_dim, out_dim), requires_grad=True))
    nn.init.xavier_uniform_(self.weight)

  def forward(self, x: torch.Tensor):
    weight = torch.unsqueeze(self.weight, 0)
    weight = weight.expand(x.shape[0], -1, -1)
    x = torch.bmm(x, weight)
    return x

class UpSample(DownSample):
  def __init__(self, in_dim, out_dim):
    super().__init__(in_dim, out_dim)


class PatchRecovery(nn.Module):
  def __init__(self, patch_size, dim):
    '''Patch recovery operation'''
    super().__init__()
    self.patch_size = patch_size
    self.conv_surface = nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=(4, 2), stride=(4, 2))
  

  def forward(self, x) -> Tuple[Tensor, Tensor]:
    x = torch.reshape(x, shape=(x.shape[0], x.shape[1], 16, 16, 8))
    bsize, length, _, _, _ = x.shape
    x = x.reshape(-1, 16, 16, 8)
    x = x.permute(0, 3, 1, 2)
    x = self.conv_surface(x)
    x = x.reshape(bsize, length, 64, 32, 8)
    return x
  

class MLP(nn.Module):
  def __init__(self, dim, dropout_rate):
    '''MLP layers, same as most vision transformer architectures.'''
    super().__init__()
    self.linear1 = nn.Linear(dim, dim * 4)
    self.linear2 = nn.Linear(dim * 4, dim)
    self.activation = nn.GELU()
    self.drop = nn.Dropout(p=dropout_rate)
    
  def forward(self, x):
    x = self.linear1(x)
    x = self.activation(x)
    x = self.drop(x)
    x = self.linear2(x)
    x = self.drop(x)
    return x
