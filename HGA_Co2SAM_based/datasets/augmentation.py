# augmentation.py
import albumentations as A


weak_transforms = A.Compose(
    [A.Flip(), A.HorizontalFlip(), A.VerticalFlip()],
    # Register additional targets to ensure geometric transforms (flip, rotation) are synchronously applied to prior maps
    additional_targets={
        'namlab': 'mask', 
        'depth': 'mask', 
        'valid_mask': 'mask'
    }
)

strong_transforms = A.Compose(
    [
        A.Posterize(),
        A.Equalize(),
        A.Sharpen(),
        A.Solarize(),
        A.RandomBrightnessContrast(),
        A.RandomShadow(),
    ]
)
