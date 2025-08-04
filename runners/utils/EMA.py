import torch
import torch.nn as nn


class EMA(nn.Module):
    def __init__(self, ema_decay=0.9999, use_num_updates=False):
        super().__init__()
        self.register_buffer('ema_decay', torch.tensor(ema_decay, dtype=torch.float32))
        self.register_buffer('num_updates', torch.tensor(0,dtype=torch.int) if use_num_updates
                             else torch.tensor(-1,dtype=torch.int))
        self.backup = {}
        self.shadow = {}

    def register(self, current_model: nn.Module):
        for name, param in current_model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def reset_device(self, current_model: nn.Module):
        for name, param in current_model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.shadow[name].to(param.data.device)

    def update(self, current_model: nn.Module, with_decay=True):
        decay = self.ema_decay
        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.ema_decay,(1 + self.num_updates) / (10 + self.num_updates))

        for name, param in current_model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                if with_decay:
                    new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                else:
                    new_average = param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, current_model: nn.Module):
        for name, param in current_model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self, current_model: nn.Module):
        for name, param in current_model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
