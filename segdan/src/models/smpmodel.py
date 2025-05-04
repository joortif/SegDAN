import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torch.optim import lr_scheduler
import time
import pandas as pd

class SMPModel(pl.LightningModule):
    def __init__(self, in_channels:int , out_classes: int, metrics, t_max, ignore_index=255, model_name="unet", encoder_name="resnet34", **kwargs):
        super().__init__()
        self.model_name = model_name.replace("-", "")
        self.encoder_name = encoder_name

        self.model = smp.create_model(
            self.model_name,
            encoder_name=self.encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

        self.evaluation_metrics = metrics
        self.ignore_index = ignore_index
        self.t_max = t_max

        # Preprocessing parameters for image normalization
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.number_of_classes = out_classes
        self.binary = out_classes == 1
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        if out_classes > 1:
            self.loss_mode = smp.losses.MULTICLASS_MODE
        else:
            self.loss_mode = smp.losses.BINARY_MODE

        self.loss_fn = smp.losses.DiceLoss(self.loss_mode, from_logits=True, ignore_index=self.ignore_index)

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        # Normalize image
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask
    
    def validate_segmentation_batch(self, image, mask, binary=False):

        assert image.ndim == 4, f"Expected image ndim=4, got {image.ndim}" # [batch_size, channels, H, W]
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0, f"Image dimensions must be divisible by 32, got {h}x{w}"


        if binary: 
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

        image, mask = self.validate_segmentation_batch(image, mask, self.binary)

        logits_mask = self.forward(image)

        logits_mask = logits_mask.contiguous()

        loss = self.loss_fn(logits_mask, mask)

        if self.binary:
            prob_mask = logits_mask.sigmoid()
            pred_mask = (prob_mask > 0.5).float()
        
        else:
            prob_mask = logits_mask.softmax(dim=1)
            pred_mask = prob_mask.argmax(dim=1)
 
        # Compute true positives, false positives, false negatives, and true negatives
        if self.binary:
            metric_args = {"mode": "binary"}
        else:
            metric_args = {"mode": "multiclass", "num_classes": self.number_of_classes}

        if self.ignore_index is not None:
            metric_args["ignore_index"] = self.ignore_index

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), **metric_args)
                
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])



        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        iou_per_class = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")
        dataset_accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
        dataset_precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
        dataset_recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")
        dataset_f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_dataset_accuracy": dataset_accuracy,
            f"{stage}_dataset_precision": dataset_precision,
            f"{stage}_dataset_recall": dataset_recall,
            f"{stage}_dataset_f1_score": dataset_f1_score,
            f"{stage}_dataset_iou": dataset_iou,
        }
        
        for class_idx, iou in enumerate(torch.mean(iou_per_class, dim=0)):
            metrics[f"{stage}_class_{class_idx}_iou"] = iou.item()


        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.t_max, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model weights saved in {path}")
        
    def save_model(self, path):
        torch.save(self.model, path)
        print(f"Complete model saved in {path}")

    def save_metrics(self, trainer, model,dataloader,experiment_name,filename="metrics.csv", mode="test",training_time=None):
        if mode == "valid":
            metrics = trainer.validate(model, dataloaders=dataloader, verbose=False)
        elif mode == "test":
            metrics = trainer.test(model, dataloaders=dataloader, verbose=False)
        else:
            raise ValueError("Mode must be 'valid' or 'test'.")

        if not metrics:
            print("No metrics to save.")
            return metrics

        df = pd.DataFrame(metrics)
        df.insert(0, "Experiment", experiment_name)

        if training_time is not None:
            df["Training Time (min)"] = round(training_time / 60.0, 2)

        if os.path.exists(filename):
            df_existing = pd.read_csv(filename, sep=';')
            df_combined = pd.concat([df_existing, df], ignore_index=True)
        else:
            df_combined = df

        df_combined.to_csv(filename, sep=';', index=False)
        print(f"Métricas guardadas en {filename}")
        return metrics

    def train(self, epochs, train_loader, valid_loader, test_loader, output_metrics_path):
        trainer = pl.Trainer(max_epochs=epochs, log_every_n_steps=1)
        start_time = time.time()
        trainer.fit(self, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total training time: {total_time / 60:.2f} minutes")

        if valid_loader is not None:
            valid_metrics = trainer.validate(self, dataloaders=valid_loader, verbose=False)
            print(valid_metrics)

        # Evaluación en test y guardar métricas 
        test_metrics = self.save_metrics(trainer, self, test_loader, f"{self.model_name} - {self.encoder_name}", output_metrics_path, mode="test", training_time=total_time)
