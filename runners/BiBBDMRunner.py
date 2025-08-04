import os
import torch

from PIL import Image
from tqdm.autonotebook import tqdm

from .utils import (
    get_optimizer,
    weights_init,
    make_dir,
    get_image_grid,
    save_single_image,
)
from .BaseRunner import BaseRunner
from utils.register import Registers
from model.BrownianBridge.BiBBDM import BiBBDM


@Registers.runners.register_with_name('BiBBDMRunner')
class BiBBDMRunner(BaseRunner):
    def __init__(self, config):
        super().__init__(config)


    def initialize_model(self, config):
        bbdmnet = BiBBDM(config.model).to(config.training.device)
        bbdmnet.apply(weights_init)
        return bbdmnet


    def initialize_optimizer_scheduler(self, dl_model, config):
        optimizer = get_optimizer(config.model.optimizer, filter(lambda p: p.requires_grad, dl_model.parameters()))
        scheduler = None
        if config.model.__contains__('lr_scheduler'):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                   mode='min',
                                                                   verbose=True,
                                                                   threshold_mode='rel',
                                                                   **vars(config.model.lr_scheduler))
        return [optimizer], [scheduler]


    def print_model_summary(self, dl_model):
        def get_parameter_number(model):
            total_num = sum(p.numel() for p in model.parameters())
            trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_num, trainable_num

        total_num, trainable_num = get_parameter_number(dl_model)
        print("Total Number of parameter: %.2fM" % (total_num / 1e6))
        print("Trainable Number of parameter: %.2fM" % (trainable_num / 1e6))
    

    def get_input(self, batch, condition_key):
        additional_info = {'task': self.config.task}
        if self.config.task == 'Inpainting':
            (a, a_name), (b, b_name), mask = batch
            mask = mask.to(self.config.training.device)
            a = a.to(self.config.training.device)
            b = b.to(self.config.training.device)
            additional_info['mask'] = mask
            additional_info['a'] = a
            additional_info['b'] = b
            context = b
        elif self.config.task == 'Colorization':
            (a, a_name), (b, b_name) = batch
            a = a.to(self.config.training.device)
            b = b.to(self.config.training.device)
            context = b
        elif self.config.task == 'SemanticSynthesis':
            (a, a_name), (b, b_name), one_hot_label = batch
            a = a.to(self.config.training.device)
            b = b.to(self.config.training.device)
            one_hot_label = one_hot_label.to(self.config.training.device)
            context = one_hot_label
        else:
            (a, a_name), (b, b_name) = batch
            a = a.to(self.config.training.device)
            b = b.to(self.config.training.device)
            if self.config.args.sample_a2b and not self.config.args.sample_b2a:
                context = a
            elif self.config.args.sample_b2a and not self.config.args.sample_a2b:
                context = b
            elif self.config.args.sample_b2a and self.config.args.sample_a2b:
                context = None
            else:
                raise NotImplementedError

        if condition_key == 'nocond':
            additional_info['context'] = None
        else:
            additional_info['context'] = context
        return (a, a_name), (b, b_name), additional_info


    @torch.no_grad()
    def validation_epoch(self, val_loader, epoch, step):
        self.apply_ema()
        self.dl_model.eval()

        pbar = tqdm(val_loader, total=len(val_loader), smoothing=0.01)
        val_step = 0
        loss_sum = 0.
        additional_loss = None
        for val_batch in pbar:
            loss, additional_info = self.loss_fn(dl_model=self.dl_model,
                                batch=val_batch,
                                epoch=epoch,
                                step=val_step,
                                opt_idx=0,
                                stage='val',
                                write=False)
            if additional_loss is None:
                additional_loss = additional_info['loss']
            else:
                for key, value in additional_info['loss'].items():
                    additional_loss[key] += value
            loss_sum += loss
            val_step += 1
        average_loss = loss_sum / val_step
        self.logger['val_loss_logger'][f'{epoch}'] = {
            'epoch' : epoch,
            'total_loss' : average_loss.item(),
        }
        for key, value in additional_loss.items():
            self.logger['val_loss_logger'][f'{epoch}'][key] = value.item() / val_step
        self.writer.add_scalar(f'val_epoch/loss', average_loss, epoch)
        self.restore_ema()
        return average_loss


    def loss_fn(self, dl_model, batch, epoch, step, opt_idx=0, stage='train', write=True):
        if self.config.training.use_DDP:
            (a, a_name), (b, b_name), additional_info = self.get_input(batch, dl_model.module.condition_key)
        else:
            (a, a_name), (b, b_name), additional_info = self.get_input(batch, dl_model.condition_key)
        loss, additional_info = dl_model(a, b, context=additional_info['context'])

        if write:
            self.writer.add_scalar(f'{stage}/total_loss', loss, step)
            for key, value in additional_info["loss"].items():
                self.writer.add_scalar(f'{stage}/{key}', value, step)
        if stage == 'val':
            return loss, additional_info
        return loss
    

    @torch.no_grad()
    def sample(self, dl_model, batch, sample_path, stage='train'):
        sample_path = make_dir(os.path.join(sample_path, f'{stage}_sample'))
        
        (a, a_name), (b, b_name), additional_info = self.get_input(batch, dl_model.condition_key)
        context = additional_info['context']

        batch_size = a.shape[0] if a.shape[0] < 4 else 4
        a = a[0:batch_size]
        b = b[0:batch_size]
        context = context[0:batch_size] if context is not None else context

        grid_size = 4
        num_timesteps = self.config.model.sample_step

        # sample b2a
        if self.config.args.sample_b2a:
            b2a_sample_path = make_dir(os.path.join(sample_path, 'b2a_sample'))
            b2a_one_step_path = make_dir(os.path.join(sample_path, 'b2a_one_step_samples'))

            if self.config.args.sample_mid_steps:
                samples, one_step_samples = dl_model.b2a_sample_loop(b=b,
                                                                     context=context,
                                                                     clip_denoised=self.config.testing.clip_denoised,
                                                                     sample_mid_step=True)
                self.save_images(samples, b2a_sample_path, grid_size,
                                 save_interval=num_timesteps // 10,
                                 head_threshold=num_timesteps - 5,
                                 tail_threshold=5)
                                 # writer_tag=f'{stage}_sample' if stage != 'test' else None)

                self.save_images(one_step_samples, b2a_one_step_path, grid_size,
                                 save_interval=num_timesteps // 10,
                                 head_threshold=num_timesteps - 5,
                                 tail_threshold=5)
                                 # writer_tag=f'{stage}_one_step_sample' if stage != 'test' else None)

                sample = samples[-1]
            else:
                sample = dl_model.b2a_sample_loop(b=b,
                                                  context=context,
                                                  clip_denoised=self.config.testing.clip_denoised).to('cpu')
            
            image_grid = get_image_grid(sample, grid_size, to_normal=self.config.dataset.dataset_config.to_normal)
            im = Image.fromarray(image_grid)
            im.save(os.path.join(sample_path, 'b2a_sample.png'))
            if stage != 'test':
                self.writer.add_image(f'{stage}_b2a_sample', image_grid, self.global_iter, dataformats='HWC')
        
        # sample a2b
        if self.config.args.sample_a2b:
            a2b_sample_path = make_dir(os.path.join(sample_path, 'a2b_sample'))
            a2b_one_step_path = make_dir(os.path.join(sample_path, 'a2b_one_step_samples'))

            if self.config.args.sample_mid_steps:
                samples, one_step_samples = dl_model.a2b_sample_loop(a=a,
                                                                     context=context,
                                                                     clip_denoised=self.config.testing.clip_denoised,
                                                                     sample_mid_step=True)
                self.save_images(samples, a2b_sample_path, grid_size,
                                 save_interval=num_timesteps // 10,
                                 head_threshold=num_timesteps - 5,
                                 tail_threshold=5)
                                 # writer_tag=f'{stage}_sample' if stage != 'test' else None)

                self.save_images(one_step_samples, a2b_one_step_path, grid_size,
                                 save_interval=num_timesteps // 10,
                                 head_threshold=num_timesteps - 5,
                                 tail_threshold=5)
                                 # writer_tag=f'{stage}_one_step_sample' if stage != 'test' else None)

                sample = samples[-1]
            else:
                sample = dl_model.a2b_sample_loop(a=a,
                                                  context=context,
                                                  clip_denoised=self.config.testing.clip_denoised).to('cpu')
            
            image_grid = get_image_grid(sample, grid_size, to_normal=self.config.dataset.dataset_config.to_normal)
            im = Image.fromarray(image_grid)
            im.save(os.path.join(sample_path, 'a2b_sample.png'))
            if stage != 'test':
                self.writer.add_image(f'{stage}_a2b_sample', image_grid, self.global_iter, dataformats='HWC')
    
        image_grid = get_image_grid(b.to('cpu'), grid_size, to_normal=self.config.dataset.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'b.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_b', image_grid, self.global_iter, dataformats='HWC')

        image_grid = get_image_grid(a.to('cpu'), grid_size, to_normal=self.config.dataset.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'a.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_a', image_grid, self.global_iter, dataformats='HWC')


    @torch.no_grad()
    def sample_to_eval(self, dl_model, test_loader, sample_path):
        A_path = make_dir(os.path.join(sample_path, 'A'))
        B_path = make_dir(os.path.join(sample_path, 'B'))
        A_sample_path = make_dir(os.path.join(sample_path, f'A_{self.config.model.sample_type}_{str(self.config.model.sample_step)}'))
        B_sample_path = make_dir(os.path.join(sample_path, f'B_{self.config.model.sample_type}_{str(self.config.model.sample_step)}'))

        batch_size = self.config.dataset.test.batch_size
        to_normal = self.config.dataset.dataset_config.to_normal
        sample_num = self.config.testing.sample_num

        pbar = tqdm(test_loader, total=len(test_loader), smoothing=0.01)
        if self.config.training.use_DDP:
            test_loader.sampler.set_epoch(0)

        debug_iter_count = 0
        for test_batch in pbar:
            debug_iter_count += 1
            # if not self.config.args.debug or debug_iter_count <= 1:
            if debug_iter_count < 2:
                (a, a_name), (b, b_name), additional_info = self.get_input(test_batch, dl_model.condition_key)
                context = additional_info['context']

                for j in range(sample_num):
                    if self.config.args.sample_a2b:
                        B_sample = dl_model.a2b_sample_loop(a=a,
                                                            context=context,
                                                            clip_denoised=self.config.testing.clip_denoised).to('cpu')
                
                    if self.config.args.sample_b2a:
                        A_sample = dl_model.b2a_sample_loop(b=b,
                                                            context=context,
                                                            clip_denoised=self.config.testing.clip_denoised).to('cpu')
                    
                    for i in range(batch_size):
                        if j == 0:
                            a_gt = a[i].detach().clone()
                            b_gt = b[i].detach().clone()
                            save_single_image(a_gt, A_path, f'{a_name[i]}.png', to_normal=to_normal)
                            save_single_image(b_gt, B_path, f'{b_name[i]}.png', to_normal=to_normal)
                        
                        if self.config.args.sample_a2b:
                            B_result = B_sample[i]
                            if sample_num > 1:
                                B_sample_path_i = make_dir(os.path.join(B_sample_path, b_name[i]))
                                save_single_image(B_result, B_sample_path_i, f'output_{j}.png', to_normal=to_normal)
                            else:
                                save_single_image(B_result, B_sample_path, f'{b_name[i]}.png', to_normal=to_normal)

                        if self.config.args.sample_b2a:
                            A_result = A_sample[i]
                            if sample_num > 1:
                                A_sample_path_i = make_dir(os.path.join(A_sample_path, a_name[i]))
                                save_single_image(A_result, A_sample_path_i, f'output_{j}.png', to_normal=to_normal)
                            else:
                                save_single_image(A_result, A_sample_path, f'{a_name[i]}.png', to_normal=to_normal)
    
    @torch.no_grad()
    def sample_single_image(self, dl_model, img_path, save_path):
        from datasets.utils import get_image_paths_from_dir, preprocess_dataset_config, read_image, transform_image
        from PIL import Image
        img, img_name = read_image(img_path)
        dataset_config = preprocess_dataset_config(self.config.dataset.dataset_config, stage='test')
        to_normal = self.config.dataset.dataset_config.to_normal
        img, _ = transform_image(dataset_config=dataset_config, image=img, flip=False)
        img = img.unsqueeze(0).to(self.config.training.device)

        if self.config.args.sample_a2b:
            B_sample = dl_model.a2b_sample_loop(a=img,
                                                context=None,
                                                clip_denoised=self.config.testing.clip_denoised).to('cpu')
            save_single_image(B_sample[0], save_path, f'{img_name}.png', to_normal=to_normal)    
        if self.config.args.sample_b2a:
            A_sample = dl_model.b2a_sample_loop(b=img,
                                                context=None,
                                                clip_denoised=self.config.testing.clip_denoised).to('cpu')
            save_single_image(A_sample[0], save_path, f'{img_name}.png', to_normal=to_normal)

