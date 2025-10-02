import os
import logging
from typing import Optional
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torch.optim import lr_scheduler
import time
import numpy as np
import pandas as pd

from segdan.metrics.compute_metrics import compute_metrics
from segdan.models.semanticsegmentationmodel import SemanticSegmentationModel

logger = logging.getLogger(__name__)

class SMPModel(pl.LightningModule, SemanticSegmentationModel):
    def __init__(self, in_channels: int , classes: int, metrics: np.ndarray, imgsz:int, selection_metric: str, epochs:int, t_max: Optional[int], output_path:str, 
                 model_name: str="unet", encoder_name: str="resnet34", fraction:Optional[float]=0.6, **kwargs):
        
        super().__init__()

        SemanticSegmentationModel.__init__(self, classes=classes, epochs=epochs, imgsz=imgsz, metrics=metrics, 
                                           selection_metric=selection_metric, 
                                           model_name=model_name, model_size=encoder_name, output_path=output_path, fraction=fraction)
        self.model_name = model_name.replace("-", "")
        self.in_channels = in_channels

        self.model = smp.create_model(
            arch=self.model_name,
            encoder_name=self.model_size,
            in_channels=self.in_channels,
            classes=self.out_classes,
            **kwargs,
        )

        self.t_max = t_max
        
        # Preprocessing parameters for image normalization
        params = smp.encoders.get_preprocessing_params(self.model_size)
        self.number_of_classes = self.out_classes
        self.binary = self.out_classes == 1
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        if self.binary:
            self.loss_mode = smp.losses.BINARY_MODE
        else:
            self.loss_mode = smp.losses.MULTICLASS_MODE

        self.loss_fn = smp.losses.DiceLoss(self.loss_mode, from_logits=True, ignore_index=255)

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.autobatch_imgsz()

    def forward(self, image):
        # Normalize image
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask
    
    def validate_segmentation_batch(self, image, mask):

        assert image.ndim == 4, f"Expected image ndim=4, got {image.ndim}" # [batch_size, channels, H, W]
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0, f"Image dimensions must be divisible by 32, got {h}x{w}"

        if self.binary: 
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            assert mask.ndim == 4, f"Expected binary mask ndim=4, got {mask.ndim}"
            assert mask.max() <= 1.0 and mask.min() >= 0.0, "Binary mask values must be in range [0, 1]"
        else:
            assert mask.ndim == 3, f"Expected multiclass mask ndim=3, got {mask.ndim}"
            mask = mask.long()
        
        return image, mask

    def shared_step(self, batch, stage):
        image, mask = batch

        image, mask = self.validate_segmentation_batch(image, mask)
        
        logits_mask = self.forward(image)

        logits_mask = logits_mask.contiguous()

        loss = self.loss_fn(logits_mask, mask)

        if self.binary:
            prob_mask = logits_mask.sigmoid()
            pred_mask = (prob_mask > 0.5).float()
        
        else:
            prob_mask = logits_mask.softmax(dim=1)
            pred_mask = prob_mask.argmax(dim=1)

        if self.binary:
            metric_args = {"mode": "binary"}
        else:
            metric_args = {"mode": "multiclass", "num_classes": self.number_of_classes}

        metric_args["ignore_index"] = 255

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), **metric_args)
                
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        results = compute_metrics(outputs, self.metrics, self.classes, stage)

        self.log_dict(results, prog_bar=True)

    def training_step(self, batch):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.t_max, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def save_metrics(self,trainer, dataloader, experiment_name, filename, training_time=None):
        metrics = trainer.test(dataloaders=dataloader)
        if not metrics:
            logger.warning("No metrics to save.")
            return metrics

        metrics_dict = metrics[0]  

        df = pd.DataFrame([metrics_dict])  
        df.insert(0, "Experiment", experiment_name)  
        
        if training_time is not None:
            df["Training Time (min)"] = round(training_time / 60.0, 2)

        if os.path.exists(filename):
            df_existing = pd.read_csv(filename, sep=';')
            df_combined = pd.concat([df_existing, df], ignore_index=True)
        else:
            df_combined = df

        file_output_path = os.path.join(self.output_path, filename)
        df_combined.to_csv(file_output_path, sep=';', index=False)

        self.show_metrics(metrics_dict, "Test")  

        logger.info(f"Metrics saved in file {file_output_path}")

        evaluation_metric = metrics_dict.get(f"{self.selection_metric}_test")
        return evaluation_metric

    def run_training(self, train_loader, valid_loader, test_loader):
        trainer = pl.Trainer(max_epochs=self.epochs, log_every_n_steps=1)
        start_time = time.time()
        trainer.fit(self, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Total training time: {total_time / 60:.2f} minutes")

        if valid_loader is not None:
            valid_metrics = trainer.validate(self, dataloaders=valid_loader, verbose=False)
            self.show_metrics(valid_metrics[0], "Validation")

        evaluation_metric = self.save_metrics(trainer, test_loader, f"{self.model_name} - {self.model_size}", f"metrics_{self.model_name}.csv", training_time=total_time)
        model_output_path = self.save_model(self.output_path, weights_only=False)

        return evaluation_metric, model_output_path