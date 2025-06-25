import albumentations as A

def get_training_augmentation(resize_height = 512, resize_width=512):
    train_transform = [
        A.Resize(height=resize_height, width=resize_width),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, fill=0, fill_mask=255, border_mode=0
        ),
        A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32, fill=0, fill_mask=255),
        A.RandomCrop(height=128, width=128, p=0.5),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation(resize_height = 512, resize_width=512):
    test_transform = [
       A.Resize(height=resize_height, width=resize_width),
       A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32, fill=0, fill_mask=255)
    ]
    return A.Compose(test_transform)