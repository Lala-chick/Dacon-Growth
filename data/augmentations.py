import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

train_private_transform = A.Compose([
                            A.RandomBrightness(p=0.3),
                            A.OneOf([
                                    A.OpticalDistortion(),
                                    A.GridDistortion(),
                            ]),
                            A.OneOf([
                                    A.GlassBlur(),
                                    A.MotionBlur(),
                                    A.MedianBlur(),
                                    A.GaussianBlur(),
                            ]),
                            A.GaussNoise(p=0.3),
                            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, min_height=2, min_width=2)
                            ])

train_global_transform = A.Compose([
                            A.HorizontalFlip(),
                            A.VerticalFlip(),
                            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ], additional_targets={'image1': 'image'})

test_global_transform = A.Compose([
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ], additional_targets={'image1': 'image'})