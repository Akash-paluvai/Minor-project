"""
Image transforms for training, validation, and self-supervised pretraining.
"""

from torchvision import transforms


def get_train_transforms(image_size: int = 224):
    """Standard training transforms with augmentation."""
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_val_transforms(image_size: int = 224):
    """Validation / test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_simclr_transforms(image_size: int = 224):
    """
    SimCLR-style augmentations for self-supervised contrastive pretraining.
    Returns a transform that produces TWO augmented views of the same image.
    """
    base_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return DualViewTransform(base_transform)


class DualViewTransform:
    """Apply a transform twice to produce two augmented views."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, image):
        view1 = self.base_transform(image)
        view2 = self.base_transform(image)
        return view1, view2
