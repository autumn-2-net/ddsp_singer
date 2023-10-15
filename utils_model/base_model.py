from typing import Dict
from torchmetrics import Metric, MeanMetric
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_debug, rank_zero_info, rank_zero_only

import lightning as PL

from train_utils.ssvv_BatchSampler import ssvvsc_BatchSampler, ssvvsc_BatchSampler_val


class BaseTask(PL.LightningModule):
    """
        Base class for training tasks.
        1. *load_ckpt*:
            load checkpoint;
        2. *training_step*:
            record and log the loss;
        3. *optimizer_step*:
            run backwards step;
        4. *start*:
            load training configs, backup code, log to tensorboard, start training;
        5. *configure_ddp* and *init_ddp_connection*:
            start parallel training.

        Subclasses should define:
        1. *build_model*, *build_optimizer*, *build_scheduler*:
            how to build the model, the optimizer and the training scheduler;
        2. *_training_step*:
            one training step of the model;
        3. *on_validation_end* and *_on_validation_end*:
            postprocess the validation output.
    """

    def __init__(self,config, *args, **kwargs):
        # dataset configs
        super().__init__(*args, **kwargs)

        self.config=config

        #
        # self.dataset_cls = None
        # self.max_batch_frames = self.config ['max_batch_frames']
        # self.max_batch_size = self.config ['max_batch_size']
        # self.max_val_batch_frames = self.config ['max_val_batch_frames']
        # if self.max_val_batch_frames == -1:
        #     self.config ['max_val_batch_frames'] = self.max_val_batch_frames = self.max_batch_frames
        # self.max_val_batch_size = self.config ['max_val_batch_size']
        # if self.max_val_batch_size == -1:
        #     self.config ['max_val_batch_size'] = self.max_val_batch_size = self.max_batch_size

        self.training_sampler = None
        self.model = None
        self.skip_immediate_validation = False
        self.skip_immediate_ckpt_save = False

        self.valid_losses: Dict[str, Metric] = {
            'total_loss': MeanMetric()
        }
        self.valid_metric_names = set()
        self.ssx = 0

    ###########
    # Training, validation and testing
    ###########
    # def setup(self, stage):
    #     self.phone_encoder = self.build_phone_encoder()
    #     self.model = self.build_model()
    #     # utils.load_warp(self)
    #     self.unfreeze_all_params()
    #     if self.config['freezing_enabled']:
    #         self.freeze_params()
    #     if self.config['finetune_enabled'] and get_latest_checkpoint_path(pathlib.Path(hparams['work_dir'])) is None:
    #         self.load_finetune_ckpt(self.load_pre_train_model())
    #     self.print_arch()
    #     self.build_losses_and_metrics()
    #     self.train_dataset = self.dataset_cls(hparams['train_set_name'])
    #     self.valid_dataset = self.dataset_cls(hparams['valid_set_name'])









    # @staticmethod
    # def build_phone_encoder():
    #     phone_list = build_phoneme_list()
    #     return TokenTextEncoder(vocab_list=phone_list)

    # def build_model(self):
    #     raise NotImplementedError()

    # @rank_zero_only
    # def print_arch(self):
    #     utils.print_arch(self.model)

    def build_losses_and_metrics(self):
        raise NotImplementedError()

    def register_metric(self, name: str, metric: Metric):
        assert isinstance(metric, Metric)
        setattr(self, name, metric)
        self.valid_metric_names.add(name)

    def run_model(self, sample, infer=False):
        """
        steps:
            1. run the full model
            2. calculate losses if not infer
        """
        raise NotImplementedError()

    def on_train_epoch_start(self):
        if self.training_sampler is not None:
            self.training_sampler.set_epoch(self.current_epoch)

    def _training_step(self, sample):
        """
        :return: total loss: torch.Tensor, loss_log: dict, other_log: dict
        """
        losses = self.run_model(sample)
        total_loss = sum(losses.values())
        # return total_loss, {**losses, 'step':int(self.global_step)}
        return total_loss, {**losses, }

    def training_step(self, sample, batch_idx, optimizer_idx=-1):
        total_loss, log_outputs = self._training_step(sample)

        # logs to progress bar
        self.log_dict(log_outputs, prog_bar=True, logger=False, on_step=True, on_epoch=False)
        self.log('lr', self.lr_schedulers().get_last_lr()[0], prog_bar=True, logger=False, on_step=True, on_epoch=False)
        self.log('step', int(self.global_step), prog_bar=True, logger=False, on_step=True, on_epoch=False)
        # logs to tensorboard
        if self.global_step % self.config['log_interval'] == 0:
            tb_log = {f'training/{k}': v for k, v in log_outputs.items()}
            tb_log['training/lr'] = self.lr_schedulers().get_last_lr()[0]
            self.logger.log_metrics(tb_log, step=self.global_step)

        return total_loss

    # def on_before_optimizer_step(self, *args, **kwargs):
    #     self.log_dict(grad_norm(self, norm_type=2))

    def _on_validation_start(self):
        pass

    def on_validation_start(self):
        self._on_validation_start()
        for metric in self.valid_losses.values():
            metric.to(self.device)
            metric.reset()

    def _validation_step(self, sample, batch_idx):
        """

        :param sample:
        :param batch_idx:
        :return: loss_log: dict, weight: int
        """
        raise NotImplementedError()

    def validation_step(self, sample, batch_idx):
        """

        :param sample:
        :param batch_idx:
        """
        if self.ssx ==0 and self.global_step!=0:
            self.skip_immediate_validation=True
            self.ssx=1
        if self.global_step==0:
            self.ssx = 1

        if self.skip_immediate_validation:
            rank_zero_debug(f"Skip validation {batch_idx}")
            return {}
        with torch.autocast(self.device.type, enabled=False):
            losses, weight = self._validation_step(sample, batch_idx)
        losses = {
            'total_loss': sum(losses.values()),
            **losses
        }
        for k, v in losses.items():
            if k not in self.valid_losses:
                self.valid_losses[k] = MeanMetric().to(self.device)
            self.valid_losses[k].update(v, weight=weight)
        return losses

    def on_validation_epoch_end(self):
        if self.skip_immediate_validation:
            self.skip_immediate_validation = False
            self.skip_immediate_ckpt_save = True
            return
        loss_vals = {k: v.compute() for k, v in self.valid_losses.items()}
        self.log('val_loss', loss_vals['total_loss'], on_epoch=True, prog_bar=True, logger=False, sync_dist=True)
        self.logger.log_metrics({f'validation/{k}': v for k, v in loss_vals.items()}, step=self.global_step)
        for metric in self.valid_losses.values():
            metric.reset()
        metric_vals = {k: getattr(self, k).compute() for k in self.valid_metric_names}
        self.logger.log_metrics({f'metrics/{k}': v for k, v in metric_vals.items()}, step=self.global_step)
        for metric_name in self.valid_metric_names:
            getattr(self, metric_name).reset()

    # noinspection PyMethodMayBeStatic
    def build_scheduler(self, optimizer):
        from utils.cls_loader import build_lr_scheduler_from_config

        scheduler_args = self.config['lr_scheduler_args']
        assert scheduler_args['scheduler_cls'] != ''
        scheduler = build_lr_scheduler_from_config(optimizer, scheduler_args)
        return scheduler

    # noinspection PyMethodMayBeStatic
    def build_optimizer(self, model):
        from utils.cls_loader import build_object_from_class_name_opt

        optimizer_args = self.config['optimizer_args']
        assert optimizer_args['optimizer_cls'] != ''
        if 'beta1' in optimizer_args and 'beta2' in optimizer_args and 'betas' not in optimizer_args:
            optimizer_args['betas'] = (optimizer_args['beta1'], optimizer_args['beta2'])
        optimizer = build_object_from_class_name_opt(
            optimizer_args['optimizer_cls'],
            torch.optim.Optimizer,
            model.parameters(),
            **optimizer_args
        )
        return optimizer

    def configure_optimizers(self):
        optm = self.build_optimizer(self)
        scheduler = self.build_scheduler(optm)
        if scheduler is None:
            return optm
        return {
            "optimizer": optm,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }


    def train_dataloader(self):
        self.training_sampler = ssvvsc_BatchSampler(
            self.train_dataset, batch_size=self.config['batch_size'], svs_batch_size=self.config['svs_batch_size'],


            num_replicas=(self.trainer.distributed_sampler_kwargs or {}).get('num_replicas', 1),
            rank=(self.trainer.distributed_sampler_kwargs or {}).get('rank', 0),

            shuffle=True,

            seed=self.config['seed'],drop_last=False
        )
        # self.train_dataset_collates=collates
        # self.valid_dataset_collates = collates
        return torch.utils.data.DataLoader(self.train_dataset,
                                           collate_fn=self.train_dataset_collates,
                                           batch_sampler=self.training_sampler,
                                           num_workers=self.config['DL_workers'],
                                           prefetch_factor=self.config['dataloader_prefetch_factor'],
                                           pin_memory=True,
                                           persistent_workers=True)

    def val_dataloader(self):
        sampler = ssvvsc_BatchSampler_val(
            self.valid_dataset,

            rank=(self.trainer.distributed_sampler_kwargs or {}).get('rank', 0),

        )
        return torch.utils.data.DataLoader(self.valid_dataset,
                                           collate_fn=self.valid_dataset_collates ,
                                           batch_sampler=sampler,
                                           num_workers=self.config['DL_workers_val'],
                                           prefetch_factor=self.config['dataloader_prefetch_factor'],
                                           shuffle=False)

    # def test_dataloader(self):
    #     return self.val_dataloader()
    #
    # def on_test_start(self):
    #     self.on_validation_start()
    #
    # def test_step(self, sample, batch_idx):
    #     return self.validation_step(sample, batch_idx)
    #
    # def on_test_end(self):
    #     return self.on_validation_end()

    ###########
    # Running configuration
    ###########


    # def on_save_checkpoint(self, checkpoint):
    #     # if isinstance(self.model, CategorizedModule):
    #     #     checkpoint['category'] = self.model.category
    #     checkpoint['trainer_stage'] = self.trainer.state.stage.value

    # def on_load_checkpoint(self, checkpoint):
    #     from lightning.pytorch.trainer.states import RunningStage
    #     from utils import simulate_lr_scheduler
    #     if checkpoint.get('trainer_stage', '') == RunningStage.VALIDATING.value:
    #         self.skip_immediate_validation = True
    #
    #     optimizer_args = hparams['optimizer_args']
    #     scheduler_args = hparams['lr_scheduler_args']
    #
    #     if 'beta1' in optimizer_args and 'beta2' in optimizer_args and 'betas' not in optimizer_args:
    #         optimizer_args['betas'] = (optimizer_args['beta1'], optimizer_args['beta2'])
    #
    #     if checkpoint.get('optimizer_states', None):
    #         opt_states = checkpoint['optimizer_states']
    #         assert len(opt_states) == 1  # only support one optimizer
    #         opt_state = opt_states[0]
    #         for param_group in opt_state['param_groups']:
    #             for k, v in optimizer_args.items():
    #                 if k in param_group and param_group[k] != v:
    #                     if 'lr_schedulers' in checkpoint and checkpoint['lr_schedulers'] and k == 'lr':
    #                         continue
    #                     rank_zero_info(f'| Overriding optimizer parameter {k} from checkpoint: {param_group[k]} -> {v}')
    #                     param_group[k] = v
    #             if 'initial_lr' in param_group and param_group['initial_lr'] != optimizer_args['lr']:
    #                 rank_zero_info(
    #                     f'| Overriding optimizer parameter initial_lr from checkpoint: {param_group["initial_lr"]} -> {optimizer_args["lr"]}'
    #                 )
    #                 param_group['initial_lr'] = optimizer_args['lr']
    #
    #     if checkpoint.get('lr_schedulers', None):
    #         assert checkpoint.get('optimizer_states', False)
    #         assert len(checkpoint['lr_schedulers']) == 1  # only support one scheduler
    #         checkpoint['lr_schedulers'][0] = simulate_lr_scheduler(
    #             optimizer_args, scheduler_args,
    #             step_count=checkpoint['global_step'],
    #             num_param_groups=len(checkpoint['optimizer_states'][0]['param_groups'])
    #         )
    #         for param_group, new_lr in zip(
    #             checkpoint['optimizer_states'][0]['param_groups'],
    #             checkpoint['lr_schedulers'][0]['_last_lr'],
    #         ):
    #             if param_group['lr'] != new_lr:
    #                 rank_zero_info(f'| Overriding optimizer parameter lr from checkpoint: {param_group["lr"]} -> {new_lr}')
    #                 param_group['lr'] = new_lr