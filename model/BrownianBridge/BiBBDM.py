
import torch
import itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from tqdm.autonotebook import tqdm

from .utils import set_require_grad_false, extract, default
from utils import namespace2dict, instantiate_from_config


class BiBBDM(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.model_config = model_config
        
        # Brownian Bridge parameters
        self.num_timesteps = model_config.num_timesteps
        self.mt_type = model_config.mt_type
        self.m0 = model_config.m0
        self.mT = model_config.mT
        self.eta = model_config.eta
        self.var_scale = model_config.var_scale
        
        # Sampling parameters
        self.skip_sample = model_config.skip_sample
        self.sample_type = model_config.sample_type
        self.sample_step = model_config.sample_step
        self.sample_step_type = model_config.sample_step_type
        self.steps = None

        # register hyperparameter schedule
        self.register_schedule()

        # loss and objective
        self.loss_type = model_config.loss_type
        self.objective = model_config.objective
        self.weight_obj = model_config.weight_obj
        self.weight_a_recon = model_config.weight_a_recon
        self.weight_b_recon = model_config.weight_b_recon

        # AutoEncoder
        self.autoencoder = set_require_grad_false(instantiate_from_config(namespace2dict(model_config.AutoEncoder))) if model_config.__contains__('AutoEncoder') else None

        # condition stage model
        self.condition_key = model_config.UNet.params.condition_key
        if self.condition_key == 'nocond':
            self.cond_stage_model = None
        elif self.condition_key == 'identity':
            self.cond_stage_model = torch.nn.Identity()
        elif self.condition_key == 'first_stage':
            self.cond_stage_model = self.autoencoder
        elif self.condition_key == 'trainable':
            if model_config.__contains__('cond_stage_model'):
                self.cond_stage_model = instantiate_from_config(namespace2dict(model_config.cond_stage_model))
                print(f'load cond stage model from {model_config.cond_stage_model.target}')
            else:
                self.cond_stage_model = None
        else:
            raise NotImplementedError
    
        # UNet
        self.denoise_fn = instantiate_from_config(namespace2dict(model_config.UNet))


    def register_schedule(self):
        T = self.num_timesteps

        if self.mt_type == 'linear':
            m_min, m_max = self.m0, self.mT
            m_t = np.linspace(m_min, m_max, T)
        elif self.mt_type == 'sin':
            m_t = np.arange(T) / T
            m_t[0] = 0.0005
            m_t = 0.5 * np.sin(np.pi * (m_t - 0.5)) + 0.5
        elif self.mt_type == 'log':
            steps = np.arange(T)
            head = np.exp(np.linspace(np.log(self.m0), np.log(0.1), 270))
            mid = np.linspace(0.10165, 0.89835, 460)
            tail = np.flip(1. - head)
            m_t = np.concatenate((head, mid, tail))
        else:
            raise NotImplementedError
        variance_t = (m_t - m_t ** 2) * self.var_scale

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('variance_t', to_torch(variance_t))

        if self.skip_sample:
            if self.sample_step_type == 'linear':
                midsteps = torch.arange(self.num_timesteps - 2, 1,
                                        step=-((self.num_timesteps - 3) / (self.sample_step - 3))).long()
                self.steps = torch.cat((torch.Tensor([self.num_timesteps - 1]).long(),
                                        midsteps, torch.Tensor([1, 0]).long()), dim=0)
            elif self.sample_step_type == 'cosine':
                steps = np.linspace(start=0, stop=self.num_timesteps, num=self.sample_step + 1)
                steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.) / 2. * self.num_timesteps
                self.steps = torch.from_numpy(steps)
        else:
            self.steps = torch.arange(self.num_timesteps - 1, -1, -1)
    

    def get_ema_net(self):
        return self.denoise_fn
    

    def apply(self, weight_init):
        self.denoise_fn.apply(weight_init)
        if self.condition_key == 'trainable':
            self.cond_stage_model.apply(weight_init)
        return self
    

    def get_parameters(self):
        if self.condition_key == 'trainable':
            print("get parameters to optimize: Cond Stage Model, UNet")
            params = itertools.chain(self.denoise_fn.parameters(), 
                                     self.cond_stage_model.parameters())
        else:
            print("get parameters to optimize: UNet")
            params = self.denoise_fn.parameters()
        return params
    

    def get_cond_stage_context(self, input):
        if input is not None and self.cond_stage_model is not None:
            context = self.cond_stage_model(input)
            if self.condition_key != 'trainable':
                context = context.detach()
        else:
            context = None
        return context
    

    @torch.no_grad()
    def encode(self, x):
        if self.autoencoder is None:
            return x
        else:
            return self.autoencoder.encode(x).detach()


    @torch.no_grad()
    def decode(self, x_latent):
        if self.autoencoder is None:
            return x_latent
        else:
            return self.autoencoder.decode(x_latent).detach()
        

    def forward(self, a, b, context=None):
        with torch.no_grad():
            a_latent = self.encode(a)
            b_latent = self.encode(b)
        context = self.get_cond_stage_context(context)
        t = torch.randint(0, self.num_timesteps, (a.shape[0],), device=a.device).long()
        return self.p_losses(a_latent, b_latent, context, t)
    

    def p_losses(self, a, b, context, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(a))
        
        x_t, objective = self.q_sample(a, b, t, noise)
        objective_recon = self.denoise_fn(x_t, t, context=context)

        bs, c, h, w = objective_recon.shape
        a_recon = a if self.objective == 'gradb' or self.objective == 'b' \
            else self.predict_a_from_objective(x_t, b, t, objective_recon)
        b_recon = b if self.objective == 'grada' or self.objective == 'a' \
            else self.predict_b_from_objective(x_t, a, t, objective_recon)

        if self.loss_type == 'l1':
            obj_rec_loss = (objective - objective_recon).abs().mean()
            obj_rec_loss_1 = obj_rec_loss_2 = obj_rec_loss
            if self.objective == 'dlns' or self.objective == 'dlab' or self.objective == 'dlgab':
                obj_rec_loss_1 = (objective[:, 0:c//2, :, :] - objective_recon[:, 0:c//2, :, :]).abs().mean()
                obj_rec_loss_2 = (objective[:, c//2:, :, :] - objective_recon[:, c//2:, :, :]).abs().mean()
            a_rec_loss = (a - a_recon).abs().mean()
            b_rec_loss = (b - b_recon).abs().mean()
        elif self.loss_type == 'l2':
            obj_rec_loss = F.mse_loss(objective, objective_recon)
            obj_rec_loss_1 = obj_rec_loss_2 = obj_rec_loss
            if self.objective == 'dlns' or self.objective == 'dlab' or self.objective == 'dlgab':
                obj_rec_loss_1 = F.mse_loss(objective[:, 0:c//2, :, :], objective_recon[:, 0:c//2, :, :])
                obj_rec_loss_2 = F.mse_loss(objective[:, c//2:, :, :], objective_recon[:, c//2:, :, :])
            a_rec_loss = F.mse_loss(a, a_recon)
            b_rec_loss = F.mse_loss(b, b_recon)
        else:
            raise NotImplementedError
        
        loss = self.weight_obj * obj_rec_loss + self.weight_a_recon * a_rec_loss + self.weight_b_recon * b_rec_loss
        log_dict = {
            "loss": {
                'a_rec_loss': a_rec_loss,
                'b_rec_loss': b_rec_loss,
                'obj_rec_loss': obj_rec_loss,
                'obj_rec_loss_1': obj_rec_loss_1,
                'obj_rec_loss_2': obj_rec_loss_2
            },
            "a_recon": self.decode(a_recon),
            "b_recon": self.decode(b_recon)
        }
        return loss, log_dict
    

    def q_sample(self, a, b, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(a))
        m_t = extract(self.m_t, t, a.shape)
        var_t = extract(self.variance_t, t, a.shape)
        sigma_t = torch.sqrt(var_t)

        if self.objective == 'a':
            objective = a
        elif self.objective == 'grada':
            objective = m_t * (b - a) + sigma_t * noise
        elif self.objective == 'b':
            objective = b
        elif self.objective == 'gradb':
            objective = (m_t - 1.) * (b - a) + sigma_t * noise
        elif self.objective == 'noise':
            objective = noise
        elif self.objective == 'bsuba':
            objective = b - a
        elif self.objective == 'dlns':
            objective = torch.cat((b-a, noise), dim=1)
        elif self.objective == 'dlab':
            objective = torch.cat((a, b), dim=1)
        elif self.objective == 'dlgab':
            objective_a = m_t * (b - a) + sigma_t * noise
            objective_b = (m_t - 1.) * (b - a) + sigma_t * noise
            objective = torch.cat((objective_a, objective_b), dim=1)
        else:
            raise NotImplementedError
    
        return (
            (1. - m_t) * a + m_t * b + sigma_t * noise,
            objective
        )
    

    def predict_a_from_objective(self, x_t, b, t, objective_recon):
        _, c, _, _ = objective_recon.shape
        if self.objective == 'a':
            a_recon = objective_recon
        elif self.objective == 'grada':
            a_recon = x_t - objective_recon
        elif self.objective == 'noise':
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            a_recon = (x_t - m_t * b - sigma_t * objective_recon) / (1. - m_t + 1.e-8)
        elif self.objective == 'bsuba':
            a_recon = b - objective_recon
        elif self.objective == 'dlns':
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            bsuba_recon = objective_recon[:, 0:c//2, :, :]
            noise_recon = objective_recon[:, c//2:, :, :]
            a_recon = x_t - m_t * bsuba_recon - sigma_t * noise_recon
        elif self.objective == 'dlab':
            a_recon = objective_recon[:, 0:c//2, :, :]
        elif self.objective == 'dlgab':
            a_recon = x_t - objective_recon[:, 0:c//2, :, :]
        else:
            raise NotImplementedError
        return a_recon


    def predict_b_from_objective(self, x_t, a, t, objective_recon):
        _, c, _, _ = objective_recon.shape
        if self.objective == 'b':
            b_recon = objective_recon
        elif self.objective == 'gradb':
            b_recon = x_t - objective_recon
        elif self.objective == 'noise':
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            b_recon = (x_t - (1. - m_t) * a - sigma_t * objective_recon) / (m_t + 1.e-8)
        elif self.objective == 'bsuba':
            b_recon = a + objective_recon
        elif self.objective == 'dlns':
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            bsuba_recon = objective_recon[:, 0:c//2, :, :]
            noise_recon = objective_recon[:, c//2:, :, :]
            b_recon = x_t - (m_t - 1.) * bsuba_recon - sigma_t * noise_recon
        elif self.objective == 'dlab':
            b_recon = objective_recon[:, c//2:, :, :]
        elif self.objective == 'dlgab':
            b_recon = x_t - objective_recon[:, c//2:, :, :]
        else:
            raise NotImplementedError
        return b_recon
    

    @torch.no_grad()
    def b2a_sample_loop(self, b, context=None, clip_denoised=False, sample_mid_step=False):
        context = self.get_cond_stage_context(context)
        with torch.no_grad():
            b = self.encode(b)

        # img = b + torch.randn_like(b) * torch.sqrt(self.variance_t[-1])
        img = b
        if sample_mid_step:
            imgs, one_step_imgs = [img], []
            for i in tqdm(range(len(self.steps)), desc=f'B2A sampling loop time step', total=len(self.steps)):
                img, a_recon = self.b2a_sample_step(x_t=imgs[-1],
                                                    b=b,
                                                    context=context,
                                                    i=i,
                                                    clip_denoised=clip_denoised)
                one_step_imgs.append(a_recon)
            
            for i in range(len(imgs)):
                imgs[i] = self.decode(imgs[i])
            for i in range(len(one_step_imgs)):
                one_step_imgs = self.decode(one_step_imgs[i])

            return imgs, one_step_imgs
        else:
            for i in tqdm(range(len(self.steps)), desc=f'B2A sampling loop time step', total=len(self.steps)):
                img, _ = self.b2a_sample_step(x_t=img,
                                              b=b,
                                              context=context,
                                              i=i,
                                              clip_denoised=clip_denoised)
            return self.decode(img)
        
    
    @torch.no_grad()
    def b2a_sample_step(self, x_t, b, context, i, clip_denoised=False):
        bs, *_, device = *x_t.shape, x_t.device

        # common variables
        t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
        objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)
        a_recon = self.predict_a_from_objective(x_t, b, t, objective_recon=objective_recon)
        if clip_denoised:
            a_recon.clamp_(-1., 1.)
        
        if self.steps[i] == 0: # special case: the last step
            return a_recon, a_recon
        else:
            n_t = torch.full((x_t.shape[0],), self.steps[i + 1], device=x_t.device, dtype=torch.long)
            m_t = extract(self.m_t, t, x_t.shape)
            m_nt = extract(self.m_t, n_t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            var_nt = extract(self.variance_t, n_t, x_t.shape)

            noise = torch.randn_like(x_t)
            sigma2_t = self.var_scale * (m_t - m_nt) * m_nt / m_t

            sigma_t = torch.sqrt(sigma2_t) * self.eta
            coe_eps = torch.sqrt((var_nt - sigma_t ** 2) / var_t)
            x_tminus = (1. - m_nt) * a_recon + m_nt * b + coe_eps * (x_t - (1. - m_t) * a_recon - m_t * b) + sigma_t * noise
            return x_tminus, a_recon
        

    @torch.no_grad()
    def a2b_sample_loop(self, a, context=None, clip_denoised=False, sample_mid_step=False):
        context = self.get_cond_stage_context(context)
        with torch.no_grad():
            a = self.encode(a)
        
        # img = a + torch.randn_like(a) * torch.sqrt(self.variance_t[0])
        img = a
        if sample_mid_step:
            imgs, one_step_imgs = [img], []
            for i in tqdm(reversed(range(len(self.steps))), desc=f'A2B sampling loop time step',
                          total=len(self.steps)):
                img, b_recon = self.a2b_sample_step(x_t=imgs[-1],
                                                    a=a,
                                                    context=context,
                                                    i=i,
                                                    clip_denoised=clip_denoised)
                imgs.append(img)
                one_step_imgs.append(b_recon)
            
            for i in range(len(imgs)):
                imgs[i] = self.decode(imgs[i])
            for i in range(len(one_step_imgs)):
                one_step_imgs = self.decode(one_step_imgs[i])

            return imgs, one_step_imgs
        else:
            for i in tqdm(reversed(range(len(self.steps))), desc=f'A2B sampling loop time step',
                          total=len(self.steps)):
                img, _ = self.a2b_sample_step(x_t=img,
                                              a=a,
                                              context=context,
                                              i=i,
                                              clip_denoised=clip_denoised)
            return self.decode(img) 


    @torch.no_grad()
    def a2b_sample_step(self, x_t, a, context, i, clip_denoised=False):
        bs, *_, device = *x_t.shape, x_t.device

        # common variables
        t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
        objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)
        b_recon = self.predict_b_from_objective(x_t, a, t, objective_recon=objective_recon)
        if clip_denoised:
            b_recon.clamp_(-1., 1.)

        if self.steps[i] == self.num_timesteps - 1:  # special case: the last step
            return b_recon, b_recon
        else:
            n_t = torch.full((x_t.shape[0],), self.steps[i - 1], device=x_t.device, dtype=torch.long)
            m_t = extract(self.m_t, t, x_t.shape)
            m_nt = extract(self.m_t, n_t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            var_nt = extract(self.variance_t, n_t, x_t.shape)

            sigma2_t = 2 * (1 - m_nt) * (m_nt - m_t) / (1 - m_t)
            noise = torch.randn_like(x_t)

            sigma_t = torch.sqrt(sigma2_t) * self.eta
            coe_eps = torch.sqrt((var_nt - sigma_t ** 2) / var_t)
            x_tplus = (1. - m_nt) * a + m_nt * b_recon + coe_eps * (x_t - (1. - m_t) * a - m_t * b_recon) + sigma_t * noise
            return x_tplus, b_recon
