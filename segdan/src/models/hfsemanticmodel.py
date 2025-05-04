import time
from transformers import Mask2FormerForUniversalSegmentation, MaskFormerForInstanceSegmentation, OneFormerForUniversalSegmentation, MaskFormerImageProcessor, OneFormerImageProcessor
from transformers import TrainingArguments, Trainer

class HFFormerModel():

    MODEL_CONFIGS = {
        "maskformer": {
            "model_class": MaskFormerForInstanceSegmentation,
            "base_name": "facebook/maskformer-swin-{size}-ade",
            "processor_class": MaskFormerImageProcessor,
        },
        "mask2former": {
            "model_class": Mask2FormerForUniversalSegmentation,
            "base_name": "facebook/mask2former-swin-{size}-ade-semantic",
            "processor_class": MaskFormerImageProcessor,
        },
        "oneformer": {
            "model_class": OneFormerForUniversalSegmentation,
            "base_name": "shi-labs/oneformer_ade20k_swin_{size}",
            "processor_class": OneFormerImageProcessor,
        },
    }

    def __init__(self, model_name, model_size, out_classes, metrics, selection_metric, epochs, batch_size):
        model_name = model_name.lower()
        model_size = model_size.lower()

        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported HuggingFace semantic segmentation model {model_name}. Supported models are: {list(self.MODEL_CONFIGS.keys())}")
        
        config = self.MODEL_CONFIGS[model_name]
        pretrained_name = config["base_name"].format(size=model_size)
        
        self.model = config["model_class"].from_pretrained(pretrained_name, num_labels=out_classes, ignore_mismatched_sizes=True)
        self.feature_extractor = config["processor_class"].from_pretrained(pretrained_name, do_resize=False)

        self.metrics = metrics
        self.selection_metric = selection_metric
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, train_dataset, valid_dataset):

        training_args = TrainingArguments(
            output_dir=None,
            eval_strategy="epoch",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=2e-4, #considerar probar con default=5e-5
            num_train_epochs=self.epochs,
            weight_decay=0.01,
            logging_dir=None,
            logging_strategy="no",
            save_strategy="best"
        )

        trainer = Trainer(
            model=self.model,
            args = training_args,
            train_dataset = train_dataset,
            eval_dataset = valid_dataset
            #compute_loss_func=?????,
            #compute_metrics=????
        )

        start_time = time.time()
        trainer.train()
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total training time: {total_time / 60:.2f} minutes")
        
    



