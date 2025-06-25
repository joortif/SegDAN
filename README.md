# SegDAN

In this work, we present an AutoML framework that facilitates the construction of segmentation models implemented across multiple libraries. The framework supports users throughout the entire pipeline including the analysis and
split of datasets of images, the training of the models, and their evaluation. 

Thus, this framework will help to lower the entry barrier for applying state-of-the-art segmentation techniques.

## Architecture of the framework

The workflow is depicted in the image below and can be summarized as follows:

First, the user provides the path to a segmentation dataset, which includes a set of images and their corresponding
annotations, and some configuration parameters. 

Then, the dataset is analysed to show some statistics about it, and find possible issues (like duplicates or missing information). 

After that, the dataset is split using either a hold-out or a k-fold approach. 

Subsequently, several segmentation models are trained and compared using different metrics. 

Finally, from that comparison, the best model is selected and provided to the user. 

![Architecture of the framework](./assets/arquitecture.png)

## Compatible models

Currently, the framework is still under development, and support is limited to the [Segmentation Models](https://github.com/qubvel-org/segmentation_models.pytorch) library. The framework supports the following models for semantic segmentation:

| Model name |
|--------------|
| Unet | 
| Unet++ |
| MAnet | 
| Linknet | 
| FPN | 
| PSPNet | 
| PAN | 
| DeepLabV3 | 
| DeepLabV3+ | 
| UPerNet | 
| Segformer | 
| DPT | 

Models can be used with different encoders, that can be found in the Segmentation Models library [documentation](https://smp.readthedocs.io/en/latest/encoders.html).

## Usage

Our framework is designed to accommodate different types of users. Expert users can employ the framework as Python libraries or APIs, invoking various methods to analyse the dataset, perform data splitting, train segmentation models from
multiple libraries, and evaluate the results. In contrast, non-expert users can rely on a graphical wizard  that guides them through each step of the process, allowing them to configure and build segmentation models without
requiring programming knowledge. 

This wizard can also be employed by expert users to obtain an initial pipeline that can be later refined.
