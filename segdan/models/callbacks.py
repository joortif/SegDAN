import os
import torch
import logging

from pytorch_lightning.callbacks import Callback
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState

logger = logging.getLogger(__name__)

class SaveWeightsBase:
    def __init__(
        self,
        output_path: str,
        save_n_ckpts: int = 5,
        save_last_epochs: bool = False,
    ):
        self.save_n_ckpts = int(save_n_ckpts)
        self.save_last_epochs = bool(save_last_epochs)
        self.save_dir = os.path.join(output_path, "checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)

    def _get_model_obj(self, model):
        if hasattr(model, "module"):
            return model.module
        return model

    def _make_filename(self, base_name: str) -> str:
        return f"{base_name}.pt"
    
    def _save(self, filename: str, model):

        model_obj = self._get_model_obj(model)
        full_path = os.path.join(self.save_dir, filename)

        final_path = torch.save(model_obj.state_dict(), full_path)
        logger.info(f"Checkpoint saved: {final_path}")

        return final_path
        

class SaveWeightsCallbackPL(Callback, SaveWeightsBase):
    def __init__(self, output_path, save_n_ckpts=5, save_last_epochs=False, keep_only_model_state=True, prefix="checkpoint"):
        SaveWeightsBase.__init__(self, output_path, save_n_ckpts, save_last_epochs, keep_only_model_state, prefix=prefix)
        Callback.__init__(self)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1  
        max_epochs = getattr(trainer, "max_epochs", None)

        save_this = False
        if not self.save_last_epochs:
            if epoch % self.save_n_ckpts == 0:
                save_this = True
        else:
            if max_epochs is not None and epoch > max_epochs - self.save_n_ckpts:
                save_this = True

        if save_this:
            fname = f"epoch_{epoch}"
            if self.save_last_epochs and max_epochs is not None and epoch > max_epochs - self.save_n_ckpts:
                fname = f"{fname}_last"
            filename = f"{fname}.pt"
            model_obj = getattr(pl_module, "model", pl_module)
            self._save(filename, model_obj)


class SaveWeightsCallbackHF(TrainerCallback, SaveWeightsBase):
    def __init__(self, output_path, save_n_ckpts=5, save_last_epochs=False, prefix="checkpoint"):
        SaveWeightsBase.__init__(self, output_path, save_n_ckpts, save_last_epochs, prefix=prefix)
        TrainerCallback.__init__(self)

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """
        args: TrainingArguments
        state: TrainerState tiene 'epoch' y 'global_step' (a veces epoch puede ser float/None)
        kwargs['model'] -> modelo
        """
        model = kwargs.get("model")

        epoch = None
        if getattr(state, "epoch", None) is not None:
            try:
                epoch = int(state.epoch) + 1
            except Exception:
                epoch = None

        if epoch is None:
            epoch = getattr(state, "global_step", 0)

        total_epochs = getattr(args, "num_train_epochs", None)

        save_this = False
        if not self.save_last_epochs:
            if epoch % self.save_n_ckpts == 0:
                save_this = True
        else:
            if total_epochs is not None and epoch > total_epochs - self.save_n_ckpts:
                save_this = True

        if save_this:
            fname = f"epoch_{epoch}"
            if self.save_last_epochs and total_epochs is not None and epoch > total_epochs - self.save_n_ckpts:
                fname = f"{fname}_last"


            model_obj = self._get_model_obj(model)
            hf_name = getattr(model_obj, "name_or_path", None) or getattr(model_obj, "__class__", None).__name__
            filename = self._make_filename(f"{hf_name}_{fname}")

            extra = {"epoch": epoch}
            self._save(filename, model_obj, extra=extra)
