import argparse
import pathlib

import torch
from lightning.pytorch.loggers import TensorBoardLogger

from PL_callbacks.save_checkpoint import ModelCheckpoints
from utils.config_loader import get_config
from utils_model.ddsp_sinder_task import ddsp_ss
# from utils_model.ssvc import ssvc
# from utils_model.ssvc_v3 import ssvc
import lightning as pl
import os
# os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING']='1'
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # Prevent unacceptable slowdowns when using 16 precision


def main():
    parser = argparse.ArgumentParser(description='Train a DDSP or NSF Vocos singer')
    parser.add_argument('--config', default='configs/a_v2.yaml')
    parser.add_argument('--work-dir')
    parser.add_argument('--resume', help='Resume a checkpoint including optimizer state')
    parser.add_argument('--init-checkpoint', help='Initialize model weights only')
    args = parser.parse_args()
    if args.resume and args.init_checkpoint:
        parser.error('--resume and --init-checkpoint are mutually exclusive')
    config = get_config(args.config)
    config.update({'infer': False})
    models_ssvc = ddsp_ss(config=config)
    models_ssvc.build_losses_and_metrics()
    work_dir = pathlib.Path(args.work_dir or config.get(
        'work_dir', pathlib.Path(config['base_work_dir']) / 'ddsp_shuansa2'
    ))
    work_dir.mkdir(parents=True, exist_ok=True)
    # Save the resolved configuration alongside the checkpoint for inference.
    import yaml
    with (work_dir / 'config.yaml').open('w', encoding='utf-8') as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    trainer = pl.Trainer(
        accelerator=config['pl_trainer_accelerator'],
        devices=config['pl_trainer_devices'],
        num_nodes=config['pl_trainer_num_nodes'],
        strategy='auto',
        precision=config['pl_trainer_precision'],
        callbacks=[
            ModelCheckpoints(
                dirpath=work_dir,
                filename='model_ckpt_steps_{step}',
                auto_insert_metric_name=False,
                monitor='step',
                mode='max',
                save_last=False,
                # every_n_train_steps=hparams['val_check_interval'],
                save_top_k=config['num_ckpt_keep'],
                permanent_ckpt_start=config['permanent_ckpt_start'],
                permanent_ckpt_interval=config['permanent_ckpt_interval'],
                verbose=True
            ),
            # LearningRateMonitor(logging_interval='step'),
            # DsTQDMProgressBar(),
        ],
        logger=TensorBoardLogger(
            save_dir=str(work_dir),
            name='lightning_logs',
            version='lastest'
        ),
        gradient_clip_val=config['clip_grad_norm'],
        val_check_interval=config['val_check_interval'] * config['accumulate_grad_batches'],
        # so this is global_steps
        check_val_every_n_epoch=None,
        log_every_n_steps=1,
        max_steps=config['max_updates'],
        use_distributed_sampler=False,
        num_sanity_val_steps=config['num_sanity_val_steps'],
        accumulate_grad_batches=config['accumulate_grad_batches']
    )
    # Preserve the legacy initialization for the original DDSP configuration.
    legacy_init = (r'D:\propj\ddsp_singer\ckpt\ddsp_s8\model_ckpt_steps_63999.ckpt'
                   if config.get('backbone', 'ddsp') == 'ddsp' else None)
    init_checkpoint = args.init_checkpoint or config.get('init_checkpoint', legacy_init)
    if init_checkpoint and not args.resume:
        checkpoint = torch.load(init_checkpoint, map_location='cpu', weights_only=False)
        if models_ssvc.use_nsf_vocos:
            # Load only the generator; loss/metric buffers are not model weights.
            state = {k[len('model.'):]: v for k, v in checkpoint['state_dict'].items()
                     if k.startswith('model.')}
            models_ssvc.model.load_state_dict(state, strict=True)
        else:
            models_ssvc.load_state_dict(checkpoint['state_dict'], strict=False)
    trainer.fit(models_ssvc, ckpt_path=args.resume)


if __name__ == '__main__':
    main()
