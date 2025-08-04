import os
import yaml
import time
import torch
import datetime
import argparse

import torch.distributed as dist

from tqdm.autonotebook import tqdm
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from .utils import (
    make_save_dirs, 
    get_dataset, 
    remove_file, 
    make_dir
)
from .utils.EMA import EMA
from utils import save_to_file
from evaluation.FID import calc_FID


class BaseRunner(ABC):
    def __init__(self, config):
        self.dl_model = None  # Neural Network
        self.optimizer = None  # optimizer
        self.scheduler = None  # scheduler
        self.config = config  # config from configuration file

        # set training params
        self.global_epoch = 0  # global epoch
        self.global_iter = 0   # global iter
        # set log and save destination
        self.config.result = argparse.Namespace()
        self.config.result.log_path, \
        self.config.result.ckpt_path, \
        self.config.result.training_sample_path, \
        self.config.result.testing_sample_path, \
        self.config.result.evaluation_sample_path = make_save_dirs(
            result_path=self.config.args.result_path,
            prefix=self.config.dataset.dataset_name,
            suffix=self.config.model.model_name,
            with_time=False,
        )
        
        self.save_config()  # save configuration file
        self.writer = SummaryWriter(self.config.result.log_path)  # initialize SummaryWriter
        self.logger = {
            'train_loss_logger': {},
            'val_loss_logger': {},
            'a2b_fid_logger': {},
            'b2a_fid_logger': {}
        }

        # initialize model
        self.use_ema = False if not self.config.model.__contains__('EMA') else self.config.model.EMA.use_ema
        self.initialize_model_optimizer_scheduler_ema(self.config, is_train=self.config.args.train)


    # save configuration file
    def save_config(self):
        save_path = os.path.join(self.config.result.ckpt_path, 'config.yaml')
        save_config = self.config
        with open(save_path, 'w') as f:
            yaml.dump(save_config, f)
    

    def initialize_model_optimizer_scheduler_ema(self, config, is_train=True):
        """
        initialize model, optimizer, scheduler
        :param args: args
        :param config: config
        :param is_test: is_test
        :return: dl_model: Neural Network, nn.Module;
                 optimizer: a list of optimizers;
                 scheduler: a list of schedulers or None;
        """
        self.dl_model = self.initialize_model(config)
        if self.use_ema:
            self.ema = EMA(self.config.model.EMA.ema_decay)
            self.update_ema_interval = self.config.model.EMA.update_ema_interval
            self.start_ema_step = self.config.model.EMA.start_ema_step
            self.ema.register(self.dl_model.get_ema_net())
        
        self.load_model_from_checkpoint()
        self.print_model_summary(self.dl_model)

        if config.training.use_DDP:
            self.dl_model = DDP(self.dl_model, device_ids=[self.config.training.local_rank], output_device=self.config.training.local_rank)
        else:
            self.dl_model = self.dl_model.to(self.config.training.device)

        self.optimizer, self.scheduler = None, None
        if is_train:
            self.optimizer, self.scheduler = self.initialize_optimizer_scheduler(self.dl_model, config)
            self.load_optimizer_scheduler_from_checkpoint()
        return


    # load model, EMA, optimizer, scheduler from checkpoint
    def load_model_from_checkpoint(self):
        model_states = None
        if self.config.model.__contains__('model_load_path') and self.config.model.model_load_path is not None:
            print(f"load model {self.config.model.model_name} from {self.config.model.model_load_path}")
            model_states = torch.load(self.config.model.model_load_path, map_location='cpu')

            self.global_epoch = model_states['epoch']
            self.global_iter = model_states['iter']

            # load model
            self.dl_model.load_state_dict(model_states['model'])

            # load ema
            if self.use_ema and 'ema' in model_states:
                self.ema.shadow = model_states['ema']
                self.ema.reset_device(self.dl_model.get_ema_net())
        return
    
    
    # load optimizer, scheduler from chekcpoint
    def load_optimizer_scheduler_from_checkpoint(self):
        if self.config.model.__contains__('optim_sche_load_path') and self.config.model.optim_sche_load_path is not None:
            print(f"load optimizer scheduler from {self.config.model.optim_sche_load_path}")
            optimizer_scheduler_states = torch.load(self.config.model.optim_sche_load_path, map_location='cpu')
            for i in range(len(self.optimizer)):
                self.optimizer[i].load_state_dict(optimizer_scheduler_states['optimizer'][i])

            if self.scheduler is not None:
                for i in range(len(self.scheduler)):
                    self.scheduler[i].load_state_dict(optimizer_scheduler_states['scheduler'][i])
        return


    def get_checkpoint_states(self, stage='epoch_end'):
        optimizer_state = []
        for i in range(len(self.optimizer)):
            optimizer_state.append(self.optimizer[i].state_dict())

        scheduler_state = []
        if self.scheduler is not None:
            for i in range(len(self.scheduler)):
                if self.scheduler[i] is not None:
                    scheduler_state.append(self.scheduler[i].state_dict())

        optimizer_scheduler_states = {
            'optimizer': optimizer_state,
            'scheduler': scheduler_state
        }

        model_states = {
            'iter': self.global_iter,
            'epoch': self.global_epoch + 1
        }

        if self.config.training.use_DDP:
            model_states['model'] = self.dl_model.module.state_dict()
        else:
            model_states['model'] = self.dl_model.state_dict()

        if self.use_ema:
            model_states['ema'] = self.ema.shadow
        return model_states, optimizer_scheduler_states
    

    # EMA part
    def step_ema(self):
        with_decay = False if self.global_iter < self.start_ema_step else True
        if self.config.training.use_DDP:
            self.ema.update(self.dl_model.module.get_ema_net(), with_decay=with_decay)
        else:
            self.ema.update(self.dl_model.get_ema_net(), with_decay=with_decay)
    

    def apply_ema(self):
        if self.use_ema:
            if self.config.training.use_DDP:
                self.ema.apply_shadow(self.dl_model.module.get_ema_net())
            else:
                self.ema.apply_shadow(self.dl_model.get_ema_net())
    

    def restore_ema(self):
        if self.use_ema:
            if self.config.training.use_DDP:
                self.ema.restore(self.dl_model.module.get_ema_net())
            else:
                self.ema.restore(self.dl_model.get_ema_net())


    # Evaluation and sample part
    @torch.no_grad()
    def validation_step(self, val_batch, epoch, step):
        self.apply_ema()
        self.dl_model.eval()
        loss = self.loss_fn(dl_model=self.dl_model,
                            batch=val_batch,
                            epoch=epoch,
                            step=step,
                            opt_idx=0,
                            stage='val_step')
        if len(self.optimizer) > 1:
            loss = self.loss_fn(dl_model=self.dl_model,
                                batch=val_batch,
                                epoch=epoch,
                                step=step,
                                opt_idx=1,
                                stage='val_step')
        self.restore_ema()


    @torch.no_grad()
    def validation_epoch(self, val_loader, epoch, step):
        self.apply_ema()
        self.dl_model.eval()

        pbar = tqdm(val_loader, total=len(val_loader), smoothing=0.01)
        step = 0
        loss_sum = 0.
        dloss_sum = 0.
        for val_batch in pbar:
            loss = self.loss_fn(dl_model=self.dl_model,
                                batch=val_batch,
                                epoch=epoch,
                                step=step,
                                opt_idx=0,
                                stage='val',
                                write=False)
            loss_sum += loss
            if len(self.optimizer) > 1:
                loss = self.loss_fn(dl_model=self.dl_model,
                                    batch=val_batch,
                                    epoch=epoch,
                                    step=step,
                                    opt_idx=1,
                                    stage='val',
                                    write=False)
                dloss_sum += loss
            step += 1
        average_loss = loss_sum / step
        self.writer.add_scalar(f'val_epoch/loss', average_loss, epoch)
        if len(self.optimizer) > 1:
            average_dloss = dloss_sum / step
            self.writer.add_scalar(f'val_dloss_epoch/loss', average_dloss, epoch)
        self.restore_ema()
        return average_loss


    @torch.no_grad()
    def sample_step(self, train_batch, val_batch):
        self.apply_ema()
        self.dl_model.eval()
        sample_path = make_dir(os.path.join(self.config.result.training_sample_path, str(self.global_iter)))
        if self.config.training.use_DDP:
            self.sample(self.dl_model.module, train_batch, sample_path, stage='train')
            self.sample(self.dl_model.module, val_batch, sample_path, stage='val')
        else:
            self.sample(self.dl_model, train_batch, sample_path, stage='train')
            self.sample(self.dl_model, val_batch, sample_path, stage='val')
        self.restore_ema()


    # abstract methods
    @abstractmethod
    def print_model_summary(self, dl_model):
        pass


    @abstractmethod
    def initialize_model(self, config):
        """
        initialize model
        :param config: config
        :return: nn.Module
        """
        pass


    @abstractmethod
    def initialize_optimizer_scheduler(self, dl_model, config):
        """
        initialize optimizer and scheduler
        :param dl_model: nn.Module
        :param config: config
        :return: a list of optimizers; a list of schedulers
        """
        pass


    @abstractmethod
    def loss_fn(self, dl_model, batch, epoch, step, opt_idx=0, stage='train', write=True):
        """
        loss function
        :param dl_model: nn.Module
        :param batch: batch
        :param epoch: global epoch
        :param step: global step
        :param opt_idx: optimizer index, default is 0; set it to 1 for GAN discriminator
        :param stage: train, val, test
        :param write: write loss information to SummaryWriter
        :return: a scalar of loss
        """
        pass
    

    @abstractmethod
    def sample(self, dl_model, batch, sample_path, stage='train'):
        """
        sample a single batch
        :param dl_model: nn.Module
        :param batch: batch
        :param sample_path: path to save samples
        :param stage: train, val, test
        :return:
        """
        pass


    @abstractmethod
    def sample_to_eval(self, dl_model, test_loader, sample_path):
        """
        sample among the test dataset to calculate evaluation metrics
        :param dl_model: nn.Module
        :param test_loader: test dataloader
        :param sample_path: path to save samples
        :return:
        """
        pass


    def initialize_dataloader(self, stage):
        dataset = get_dataset(self.config.dataset, stage=stage)
        sampler = None
        if self.config.training.use_DDP:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        
        dataloader = DataLoader(dataset,
                                batch_size=self.config.dataset.train.batch_size,
                                num_workers=self.config.dataset.train.num_workers,
                                drop_last=True,
                                sampler=sampler)
        return dataloader
    

    def train(self):
        print(self.__class__.__name__)

        train_loader = self.initialize_dataloader(stage='train')
        val_loader = self.initialize_dataloader(stage='val')
        test_loader = self.initialize_dataloader(stage='test')
        print(len(train_loader))
        print(len(val_loader))
        print(len(test_loader))
        epoch_length = len(train_loader)
        start_epoch = self.global_epoch

        print(f"start training {self.config.model.model_name} on {self.config.dataset.dataset_name}, {len(train_loader)} iters per epoch")
        accumulate_grad_batches = self.config.training.accumulate_grad_batches
        train_best_a2b_fid, train_best_b2a_fid = 1000, 1000
        test_best_a2b_fid, test_best_b2a_fid = 1000, 1000
        train_best_a2b_fid_epoch, train_best_b2a_fid_epoch = 0, 0
        test_best_a2b_fid_epoch, test_best_b2a_fid_epoch = 0, 0
        for epoch in range(start_epoch, self.config.training.max_epochs):
            if self.global_iter > self.config.training.max_iters:
                break

            if self.config.training.use_DDP:
                train_loader.sampler.set_epoch(epoch)
                val_loader.sampler.set_epoch(epoch)
            
            pbar = tqdm(train_loader, total=len(train_loader), smoothing=0.01)
            self.global_epoch = epoch
            start_time = time.time()
            debug_step_count = 0
            for train_batch in pbar:
                self.global_iter += 1
                self.dl_model.train()

                debug_step_count += 1
                if not self.config.args.debug or debug_step_count <= 1:
                    losses = []
                    for i in range(len(self.optimizer)):
                        loss = self.loss_fn(dl_model=self.dl_model,
                                            batch=train_batch,
                                            epoch=epoch,
                                            step=self.global_iter,
                                            opt_idx=i,
                                            stage='train')

                        loss.backward()
                        if self.global_iter % accumulate_grad_batches == 0:
                            self.optimizer[i].step()
                            self.optimizer[i].zero_grad()
                            if self.scheduler is not None and self.scheduler[i] is not None:
                                self.scheduler[i].step(loss)
                        losses.append(loss.detach().mean()) 

                    if self.use_ema and self.global_iter % (self.update_ema_interval * accumulate_grad_batches) == 0:
                        self.step_ema()   

                    if not self.config.training.use_DDP or (self.config.training.use_DDP and self.config.training.local_rank == 0):
                        if len(self.optimizer) > 1:
                            pbar.set_description(
                                (
                                    f'Epoch: [{epoch + 1} / {self.config.training.max_epochs}] '
                                    f'iter: {self.global_iter} loss-1: {losses[0]:.4f} loss-2: {losses[1]:.4f}'
                                )
                            )
                        else:
                            pbar.set_description(
                                (
                                    f'Epoch: [{epoch + 1} / {self.config.training.max_epochs}] '
                                    f'iter: {self.global_iter} loss: {losses[0]:.4f}'
                                )
                            )
                
                    # with torch.no_grad():
                    #     if not self.config.training.use_DDP or (self.config.training.use_DDP and self.config.training.local_rank == 0):
                    #         if self.global_iter % self.config.training.validation_iter_interval == 0:
                    #             val_batch = next(iter(val_loader))
                    #             self.validation_step(val_batch=val_batch, epoch=epoch, step=self.global_iter)
                        
                    #         if self.global_iter % self.config.training.sample_iter_interval == 0:
                    #             val_batch = next(iter(val_loader))
                    #             self.sample_step(val_batch=val_batch, train_batch=train_batch)
                    #             torch.cuda.empty_cache()
                            
                            # if self.global_iter % self.config.training.save_iter_interval == 0:
                            #     print("saving latest checkpoint...")
                            #     model_states, optimizer_scheduler_states = self.get_checkpoint_states(stage='epoch_end')
                            
                            #     temp = 0
                            #     while temp < epoch + 1:
                            #         remove_file(os.path.join(self.config.result.ckpt_path, f'latest_model_{temp}.pth'))
                            #         remove_file(os.path.join(self.config.result.ckpt_path, f'latest_optim_sche_{temp}.pth'))
                            #         temp += 1
                                
                            #     torch.save(model_states, os.path.join(self.config.result.ckpt_path, f'latest_model_{epoch + 1}.pth'))
                            #     torch.save(optimizer_scheduler_states, os.path.join(self.config.result.ckpt_path, f'latest_optim_sche_{epoch + 1}.pth'))
                        
                        # if self.config.training.use_DDP:
                        #     dist.barrier()
                
            # epoch end
            end_time = time.time()
            elapsed_rounded = int(round((end_time-start_time)))
            print("training time: " + str(datetime.timedelta(seconds=elapsed_rounded)))

            if (epoch + 1) % self.config.training.save_epoch_interval == 0 \
                or (epoch + 1) == self.config.training.max_epochs \
                or self.global_iter > self.config.training.max_iters:

                if not self.config.training.use_DDP or (self.config.training.use_DDP and self.config.training.local_rank) == 0:
                    with torch.no_grad():
                        print("validating epoch...")
                        self.validation_epoch(val_loader=val_loader, epoch=epoch+1, step=self.global_iter)
                        save_to_file(self.logger['val_loss_logger'], f'{self.config.result.log_path}/val_loss_logger.yaml')
                        print("validating epoch success")
                        
                        print("inferencing samples...")
                        train_batch = next(iter(train_loader))
                        val_batch = next(iter(val_loader))
                        self.sample_step(val_batch=val_batch, train_batch=train_batch)
                        torch.cuda.empty_cache()
                        print("inferencing samples success")

                        print("saving latest checkpoint...")
                        model_states, optimizer_scheduler_states = self.get_checkpoint_states(stage='epoch_end')
                    
                        temp = 0
                        while temp < epoch + 1:
                            remove_file(os.path.join(self.config.result.ckpt_path, f'latest_model_{temp}.pth'))
                            remove_file(os.path.join(self.config.result.ckpt_path, f'latest_optim_sche_{temp}.pth'))
                            temp += 1
                        
                        torch.save(model_states, os.path.join(self.config.result.ckpt_path, f'latest_model_{epoch + 1}.pth'))
                        torch.save(optimizer_scheduler_states, os.path.join(self.config.result.ckpt_path, f'latest_optim_sche_{epoch + 1}.pth'))
                        print("saving latest checkpoint success")

            if self.config.training.use_DDP:
                dist.barrier()

            if (epoch + 1) % self.config.training.validation_epoch_interval == 0 or (epoch + 1) == self.config.training.max_epochs:
                with torch.no_grad():
                    sample_to_eval_path = self.config.result.evaluation_sample_path
                    train_eval_path = os.path.join(sample_to_eval_path, 'train')
                    test_eval_path = os.path.join(sample_to_eval_path, 'test')
                    if self.config.training.use_DDP:
                        self.sample_to_eval(self.dl_model.module, val_loader, train_eval_path)
                        self.sample_to_eval(self.dl_model.module, test_loader, test_eval_path)
                    else:
                        self.sample_to_eval(self.dl_model, val_loader, train_eval_path)
                        self.sample_to_eval(self.dl_model, test_loader, test_eval_path)
                    
                    if not self.config.training.use_DDP or (self.config.training.use_DDP and self.config.training.local_rank) == 0:
                        if self.config.args.sample_a2b:
                            train_FID_a2b = calc_FID(input_path1=os.path.join(train_eval_path, 'B'),
                                               input_path2=os.path.join(train_eval_path, f'B_{self.config.model.sample_type}_{str(self.config.model.sample_step)}'),
                                               device=self.config.training.device)
                            self.writer.add_scalar(f'val/train_FID_a2b', train_FID_a2b, epoch + 1)

                            model_states, optimizer_scheduler_states = self.get_checkpoint_states(stage='epoch_end')
                            if train_FID_a2b < train_best_a2b_fid:
                                train_best_a2b_fid, train_best_a2b_fid_epoch = train_FID_a2b.item(), epoch + 1
                                torch.save(model_states, os.path.join(self.config.result.ckpt_path, f'train_best_a2b_model.pth'))
                                torch.save(optimizer_scheduler_states, os.path.join(self.config.result.ckpt_path, f'train_best_a2b_optim_sche.pth'))
                            
                            test_FID_a2b = calc_FID(input_path1=os.path.join(test_eval_path, 'B'),
                                               input_path2=os.path.join(test_eval_path, f'B_{self.config.model.sample_type}_{str(self.config.model.sample_step)}'),
                                               device=self.config.training.device)
                            self.writer.add_scalar(f'val/test_FID_a2b', test_FID_a2b, epoch + 1)

                            model_states, optimizer_scheduler_states = self.get_checkpoint_states(stage='epoch_end')
                            if test_FID_a2b < test_best_a2b_fid:
                                test_best_a2b_fid, test_best_a2b_fid_epoch = test_FID_a2b.item(), epoch + 1
                                torch.save(model_states, os.path.join(self.config.result.ckpt_path, f'test_best_a2b_model.pth'))
                                torch.save(optimizer_scheduler_states, os.path.join(self.config.result.ckpt_path, f'test_best_a2b_optim_sche.pth'))

                            self.logger['a2b_fid_logger'][f'{epoch+1}'] = {
                                'train_best_a2b_fid' : train_best_a2b_fid,
                                'train_best_a2b_fid_epoch' : train_best_a2b_fid_epoch,
                                'test_best_a2b_fid' : test_best_a2b_fid,
                                'test_best_a2b_fid_epoch' : test_best_a2b_fid_epoch,
                                'current_epoch' : epoch + 1,
                                'current_train_a2b_fid' : train_FID_a2b.item(),
                                'current_test_a2b_fid' : test_FID_a2b.item()
                            }
                            save_to_file(self.logger['a2b_fid_logger'], f'{self.config.result.log_path}/a2b_fid_logger.yaml')

                        
                        if self.config.args.sample_b2a:
                            train_FID_b2a = calc_FID(input_path1=os.path.join(train_eval_path, 'A'),
                                               input_path2=os.path.join(train_eval_path, f'A_{self.config.model.sample_type}_{str(self.config.model.sample_step)}'),
                                               device=self.config.training.device)
                            self.writer.add_scalar(f'val/train_FID_b2a', train_FID_b2a, epoch + 1)

                            # model_states, optimizer_scheduler_states = self.get_checkpoint_states(stage='epoch_end')
                            if train_FID_b2a < train_best_b2a_fid:
                                train_best_b2a_fid, train_best_b2a_fid_epoch = train_FID_b2a.item(), epoch + 1
                                # torch.save(model_states, os.path.join(self.config.result.ckpt_path, f'best_b2a_model.pth'))
                                # torch.save(optimizer_scheduler_states, os.path.join(self.config.result.ckpt_path, f'best_b2a_optim_sche.pth'))
                            
                            test_FID_b2a = calc_FID(input_path1=os.path.join(test_eval_path, 'A'),
                                               input_path2=os.path.join(test_eval_path, f'A_{self.config.model.sample_type}_{str(self.config.model.sample_step)}'),
                                               device=self.config.training.device)
                            self.writer.add_scalar(f'val/test_FID_b2a', test_FID_b2a, epoch + 1)

                            # model_states, optimizer_scheduler_states = self.get_checkpoint_states(stage='epoch_end')
                            if test_FID_b2a < test_best_b2a_fid:
                                test_best_b2a_fid, test_best_b2a_fid_epoch = test_FID_b2a.item(), epoch + 1
                                # torch.save(model_states, os.path.join(self.config.result.ckpt_path, f'best_b2a_model.pth'))
                                # torch.save(optimizer_scheduler_states, os.path.join(self.config.result.ckpt_path, f'best_b2a_optim_sche.pth'))

                            self.logger['b2a_fid_logger'][f'{epoch+1}'] = {
                                'train_best_b2a_fid' : train_best_b2a_fid,
                                'train_best_b2a_fid_epoch' : train_best_b2a_fid_epoch,
                                'test_best_b2a_fid' : test_best_b2a_fid,
                                'test_best_b2a_fid_epoch' : test_best_b2a_fid_epoch,
                                'current_epoch' : epoch + 1,
                                'current_train_b2a_fid' : train_FID_b2a.item(),
                                'current_test_b2a_fid' : test_FID_b2a.item()
                            }
                            save_to_file(self.logger['b2a_fid_logger'], f'{self.config.result.log_path}/b2a_fid_logger.yaml')

            if self.config.training.use_DDP:
                dist.barrier()

    @torch.no_grad()
    def test_bake(self):
        test_loader = self.initialize_dataloader(stage='test')

        self.apply_ema()
        self.dl_model.eval()
        if self.config.args.sample_to_eval:
            # sample_path = self.config.result.evaluation_sample_path
            sample_path = os.path.join(self.config.result.evaluation_sample_path, 'cycle_sample')
            if self.config.training.use_DDP:
                self.sample_to_eval(self.dl_model.module, test_loader, sample_path)
            else:
                self.sample_to_eval(self.dl_model, test_loader, sample_path)
        else:
            test_iter = iter(test_loader)
            for i in tqdm(range(2), initial=0, dynamic_ncols=True, smoothing=0.01):
                test_batch = next(test_iter)
                sample_path = os.path.join(self.config.result.testing_sample_path, str(i))
                if self.config.training.use_DDP:
                    self.sample(self.dl_model.module, test_batch, sample_path, stage='test')
                else:
                    self.sample(self.dl_model, test_batch, sample_path, stage='test')
    
    @torch.no_grad()
    def test(self):
        self.calculate_average_distance()
        return

        self.apply_ema()
        self.dl_model.eval()
        
        ## b2a2b
        # image_base_path = 'results/faces2comics/LDM-L4-grada-noema/evaluation_samples/cycle_sample_b2a/A_uniform_100'
        # save_base_path = 'results/faces2comics/LDM-L4-grada-noema/evaluation_samples/cycle_sample_b2a2b/B_uniform_100'

        # image_base_path = 'results/faces2comics/LDM-L4-dlns-noema/evaluation_samples/cycle_sample_1/A_uniform_100'
        # save_base_path = 'results/faces2comics/LDM-L4-dlns-noema/evaluation_samples/cycle_sample_2/B_uniform_100'

        ## a2b2a
        # image_base_path = 'results/faces2comics/LDM-L4-gradb-noema/evaluation_samples/cycle_sample_a2b/B_uniform_100'
        # save_base_path = 'results/faces2comics/LDM-L4-gradb-noema/evaluation_samples/cycle_sample_a2b2a/A_uniform_100'

        image_base_path = 'results/faces2comics/LDM-L4-dlns-noema/evaluation_samples/cycle_sample_1/B_uniform_100'
        save_base_path = 'results/faces2comics/LDM-L4-dlns-noema/evaluation_samples/cycle_sample_2/A_uniform_100'
        
        img_dir_list = os.listdir(image_base_path)
        for dir_name in img_dir_list:
            img_dir = os.path.join(image_base_path, dir_name)
            save_path = make_dir(os.path.join(save_base_path, dir_name))

            img_list = os.listdir(img_dir)
            for img_name in img_list:
                img_path = os.path.join(img_dir, img_name)
                if self.config.training.use_DDP:
                    self.sample_single_image(self.dl_model.module, img_path, save_path)
                else:
                    self.sample_single_image(self.dl_model, img_path, save_path)

    @torch.no_grad()
    def calculate_average_distance(self):
        from datasets.utils import get_image_paths_from_dir, preprocess_dataset_config, read_image, transform_image
        from PIL import Image

        dataset_config = preprocess_dataset_config(self.config.dataset.dataset_config, stage='test')
        to_normal = self.config.dataset.dataset_config.to_normal
        
        ori_base_path = 'results/faces2comics/LDM-L4-dlns-noema/evaluation_samples/cycle_sample_1/A'
        tgt_base_path = 'results/faces2comics/LDM-L4-dlns-noema/evaluation_samples/cycle_sample_2/A_uniform_100'

        self.apply_ema()
        self.dl_model.eval()

        count = 0
        total_distance = 0
        tgt_dir_list = os.listdir(tgt_base_path)
        for dir_name in tgt_dir_list:
            ori_img_path = os.path.join(ori_base_path, f'{dir_name}.png')
            ori_img, _ = read_image(ori_img_path)
            ori_img, _ = transform_image(dataset_config=dataset_config, image=ori_img, flip=False)
            ori_img = ori_img.unsqueeze(0).to(self.config.training.device)
            ori_latent = self.dl_model.encode(ori_img)

            tgt_img_dir = os.path.join(tgt_base_path, dir_name)
            tgt_img_list = os.listdir(tgt_img_dir)
            for tgt_img_name in tgt_img_list:
                tgt_img_path = os.path.join(tgt_img_dir, tgt_img_name)
                tgt_img, _ = read_image(tgt_img_path)
                tgt_img, _ = transform_image(dataset_config=dataset_config, image=tgt_img, flip=False)
                tgt_img = tgt_img.unsqueeze(0).to(self.config.training.device)
                tgt_latent = self.dl_model.encode(tgt_img)

                count += 1
                distance = (ori_latent - tgt_latent).abs().mean().item()
                # print(distance)
                total_distance += distance
        avg_distance = total_distance / count
        print(f'average distance is {avg_distance}')
        
        var = 0
        for dir_name in tgt_dir_list:
            ori_img_path = os.path.join(ori_base_path, f'{dir_name}.png')
            ori_img, _ = read_image(ori_img_path)
            ori_img, _ = transform_image(dataset_config=dataset_config, image=ori_img, flip=False)
            ori_img = ori_img.unsqueeze(0).to(self.config.training.device)
            ori_latent = self.dl_model.encode(ori_img)

            tgt_img_dir = os.path.join(tgt_base_path, dir_name)
            tgt_img_list = os.listdir(tgt_img_dir)
            for tgt_img_name in tgt_img_list:
                tgt_img_path = os.path.join(tgt_img_dir, tgt_img_name)
                tgt_img, _ = read_image(tgt_img_path)
                tgt_img, _ = transform_image(dataset_config=dataset_config, image=tgt_img, flip=False)
                tgt_img = tgt_img.unsqueeze(0).to(self.config.training.device)
                tgt_latent = self.dl_model.encode(tgt_img)

                distance = (ori_latent - tgt_latent).abs().mean().item()
                # print(distance)
                var += (distance - avg_distance)**2
        
        var = var / count
        print(f'var is {var}')
        return avg_distance




                    
