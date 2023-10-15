import re
from typing import Dict

import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_debug, rank_zero_info, rank_zero_only
import pathlib


def get_ModelCheckpoints(config,cb_arg):
    work_dir=pathlib.Path(config['base_work_dir'])
    work_dir = work_dir/config['ckpt_name']
    return ModelCheckpoints(dirpath=work_dir,
                    filename='model_ckpt_steps_{step}',
                    auto_insert_metric_name=False,
                    monitor='step',
                    mode='max',
                    save_last=False,
                    # every_n_train_steps=hparams['val_check_interval'],
                    save_top_k=cb_arg['num_ckpt_keep'],
                    permanent_ckpt_start=cb_arg['permanent_ckpt_start'],
                    permanent_ckpt_interval=cb_arg['permanent_ckpt_interval'],
                    verbose=True)
    # return ModelCheckpoints(dirpath=work_dir,
    #                 filename='model_ckpt_steps_{step}',
    #                 auto_insert_metric_name=False,
    #                 monitor='step',
    #                 mode='max',
    #                 save_last=False,
    #                 # every_n_train_steps=hparams['val_check_interval'],
    #                 save_top_k=hparams['num_ckpt_keep'],
    #                 permanent_ckpt_start=hparams['permanent_ckpt_start'],
    #                 permanent_ckpt_interval=hparams['permanent_ckpt_interval'],
    #                 verbose=True)

class ModelCheckpoints(ModelCheckpoint):
    def __init__(
            self,
            *args,
            permanent_ckpt_start,
            permanent_ckpt_interval,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.permanent_ckpt_start = permanent_ckpt_start or 0
        self.permanent_ckpt_interval = permanent_ckpt_interval or 0
        self.enable_permanent_ckpt = self.permanent_ckpt_start > 0 and self.permanent_ckpt_interval > 9

        self._verbose = self.verbose
        self.verbose = False

    def state_dict(self):
        ret = super().state_dict()
        ret.pop('dirpath')
        return ret

    def load_state_dict(self, state_dict) -> None:
        super().load_state_dict(state_dict)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.lightning_module.skip_immediate_ckpt_save:
            trainer.lightning_module.skip_immediate_ckpt_save = False
            return
        self.last_val_step = trainer.global_step
        super().on_validation_end(trainer, pl_module)

    def _update_best_and_save(
            self, current: torch.Tensor, trainer: "pl.Trainer", monitor_candidates: Dict[str, torch.Tensor]
    ) -> None:
        k = len(self.best_k_models) + 1 if self.save_top_k == -1 else self.save_top_k

        del_filepath = None
        _op = max if self.mode == "min" else min
        while len(self.best_k_models) > k and k > 0:
            self.kth_best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
            self.kth_value = self.best_k_models[self.kth_best_model_path]

            del_filepath = self.kth_best_model_path
            self.best_k_models.pop(del_filepath)
            filepath = self._get_metric_interpolated_filepath_name(monitor_candidates, trainer, del_filepath)
            if del_filepath is not None and filepath != del_filepath:
                self._remove_checkpoint(trainer, del_filepath)

        if len(self.best_k_models) == k and k > 0:
            self.kth_best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
            self.kth_value = self.best_k_models[self.kth_best_model_path]

        super()._update_best_and_save(current, trainer, monitor_candidates)

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        filepath = (pathlib.Path(self.dirpath) / pathlib.Path(filepath).name).resolve()
        super()._save_checkpoint(trainer, str(filepath))
        if self._verbose:
            relative_path = filepath.relative_to(pathlib.Path('.').resolve())
            rank_zero_info(f'Checkpoint {relative_path} saved.')

    def _remove_checkpoint(self, trainer: "pl.Trainer", filepath: str):
        filepath = (pathlib.Path(self.dirpath) / pathlib.Path(filepath).name).resolve()
        relative_path = filepath.relative_to(pathlib.Path('.').resolve())
        search = re.search(r'steps_\d+', relative_path.stem)
        if search:
            step = int(search.group(0)[6:])
            if self.enable_permanent_ckpt and \
                    step >= self.permanent_ckpt_start and \
                    (step - self.permanent_ckpt_start) % self.permanent_ckpt_interval == 0:
                rank_zero_info(f'Checkpoint {relative_path} is now permanent.')
                return
        super()._remove_checkpoint(trainer, filepath)
        if self._verbose:
            rank_zero_info(f'Removed checkpoint {relative_path}.')
