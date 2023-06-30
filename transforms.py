import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transforms = A.Compose([
    A.HorizontalFlip(p=1),  # Horizontal flip with a probability of 1 (always)
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=1),  # Shift, scale, and rotate with given limits
    A.CoarseDropout(
        max_holes=1,
        max_height=16,
        max_width=16,
        min_holes=1,
        min_height=16,
        min_width=16,
        fill_value=np.mean((0.4914, 0.4822, 0.4465)),  # Set fill value as the mean of the dataset
        mask_fill_value=None),
        A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ToTensorV2(),
])


# Test data transformations
test_transforms = A.Compose([
        A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ToTensorV2()
                ])
