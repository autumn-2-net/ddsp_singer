import pathlib

import lightning as PL
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from utils.cls_loader import build_object_from_class_name


def load_finetune_ckpt(model, state_dict, strict_shapes: bool = True) -> None:
    adapt_shapes = strict_shapes
    if not adapt_shapes:
        cur_model_state_dict = model.state_dict()
        unmatched_keys = []
        for key, param in state_dict.items():
            if key in cur_model_state_dict:
                new_param = cur_model_state_dict[key]
                if new_param.shape != param.shape:
                    unmatched_keys.append(key)
                    print('| Unmatched keys: ', key, new_param.shape, param.shape)
        for key in unmatched_keys:
            del state_dict[key]
    model.load_state_dict(state_dict, strict=False)


def load_pre_train_model(finetune_ckpt_path: str, finetune_load_params: list):
    pre_train_ckpt_path = finetune_ckpt_path
    blacklist = finetune_load_params
    # whitelist=hparams.get('pre_train_whitelist')
    if blacklist is None:
        blacklist = []
    # if whitelist is  None:
    #     raise RuntimeError("")

    if pre_train_ckpt_path is not None:
        ckpt = torch.load(pre_train_ckpt_path)
        # if ckpt.get('category') is None:
        #     raise RuntimeError("")

        # if isinstance(self.model, CategorizedModule):
        #     self.model.check_category(ckpt.get('category'))

        state_dict = {}
        for i in ckpt['state_dict']:
            # if 'diffusion' in i:
            # if i in rrrr:
            #     continue
            skip = False
            for b in blacklist:
                if i.startswith(b):
                    skip = True
                    break

            if skip:
                continue

            state_dict[i] = ckpt['state_dict'][i]
            print(i)
        return state_dict
    else:
        raise RuntimeError("")


def get_need_freeze_state_dict_key(frozen_params, model_state_dict) -> list:
    key_list = []
    for i in frozen_params:
        for j in model_state_dict:
            if j.startswith(i):
                key_list.append(j)
    return list(set(key_list))


def freeze_params(model, frozen_params) -> None:
    model_state_dict = model.state_dict().keys()
    freeze_key = get_need_freeze_state_dict_key(frozen_params=frozen_params, model_state_dict=model_state_dict)

    for i in freeze_key:
        params = model.get_parameter(i)

        params.requires_grad = False


def unfreeze_all_params(model) -> None:
    for i in model.parameters():
        i.requires_grad = True


def build_model(config):
    model = build_object_from_class_name(cls_str=config['model_cls'], parent_cls=PL.LightningModule, strict=False,
                                         **config['model_arg'])
    if config['finetune_enabled'] and config['finetune_ckpt_path'] is not None:
        load_finetune_ckpt(model=model, state_dict=load_pre_train_model(finetune_ckpt_path=config['finetune_ckpt_path'],
                                                                        finetune_load_params=config[
                                                                            'finetune_ignored_params']),
                           strict_shapes=config['finetune_strict_shapes'])
    unfreeze_all_params(model=model)
    if config['freezing_enabled']:
        freeze_params(model=model,frozen_params=config['frozen_params'])
    return model


def build_trainer(config):
    work_dir=pathlib.Path(config['base_work_dir'])
    work_dir = work_dir/config['ckpt_name']

    if config['pl_trainer_callbacks'] is not None:
        callback = []
        for i in config['pl_trainer_callbacks']:
            cb=i['callback']
            cb_cls=cb['callback_cls']
            cb_arg = cb['callback_arg']

            callback.append(build_object_from_class_name(cls_str=cb_cls, parent_cls=None, strict=False,
                                         **{'config':config,'cb_arg':cb_arg}))

    else:
        callback = None



    trainer = PL.Trainer(
        accelerator=config['pl_trainer_accelerator'],
        devices=config['pl_trainer_devices'],
        num_nodes=config['pl_trainer_num_nodes'],
        strategy='auto',
        precision=config['pl_trainer_precision'],
        callbacks=callback,
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

    return trainer

def set_seed(config):
    if config['seed'] is not None:
        PL.seed_everything(config['seed'], workers=True)

