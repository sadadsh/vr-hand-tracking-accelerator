```
(venv) sadad-haidari@SHLinux:~/Projects/vr-hand-tracking-accelerator/vr-hand-tracking-accelerator$ source venv/bin/activate && printf "y\n" | python model/train_model.py --epochs 20 --batch_size 16 --learning_rate 0.00005 --target_accuracy 90 --dataset_root data/local_dataset/hagrid_500k --save_dir model/checkpoints_fixed | cat
CNN Training Pipeline
==================================================
Annotations: data/annotations/hagrid_all_gestures_annotations.json
Dataset root: data/local_dataset/hagrid_500k
Save directory: model/checkpoints_fixed
Target accuracy: 90.0%
Max epochs: 20
Batch size: 16
Learning rate: 5e-05
Data augmentation: True
Balanced sampling: True

Start training with fixes? (y/N): /home/sadad-haidari/Projects/vr-hand-tracking-accelerator/vr-hand-tracking-accelerator/venv/lib/python3.12/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/sadad-haidari/Projects/vr-hand-tracking-accelerator/vr-hand-tracking-accelerator/venv/lib/python3.12/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet152_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet152_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
2025-08-18 11:44:46,946 - INFO - Model initialized on cuda
2025-08-18 11:44:46,947 - INFO - Model parameters: 65,106,770
2025-08-18 11:44:46,947 - INFO - Starting training...
2025-08-18 11:44:49,559 - INFO - train split: 356518 valid samples
2025-08-18 11:44:52,191 - INFO - validation split: 101856 valid samples
2025-08-18 11:44:55,010 - INFO - test split: 50949 valid samples
2025-08-18 11:44:55,485 - INFO - Data loaders created:
2025-08-18 11:44:55,485 - INFO -   Train: 6250 batches, 356518 samples
2025-08-18 11:44:55,485 - INFO -   Val: 6366 batches, 101856 samples
2025-08-18 11:44:55,485 - INFO -   Test: 3185 batches, 50949 samples
2025-08-18 11:44:55,485 - INFO -   Effective batch size: 16
✓ Ultra-Advanced CNN created successfully
✓ Total Parameters: 65,106,770
✓ Trainable Parameters: 64,881,426
✓ Memory: 248.4 MB
✓ Backbone: ResNet152
✓ Target Accuracy: >90%
✓ Advanced Techniques: ResNet152, Multi-layer classifier, BatchNorm, Progressive dropout, Attention mechanisms

================================================================================
STARTING CNN TRAINING
================================================================================
Device: cuda
Model parameters: 65,106,770
Target accuracy: >90.0%
Max epochs: 20
Early stopping patience: 8

TRAIN Split Class Distribution:
  call           : 19,735 images
  dislike        : 19,975 images
  fist           : 19,434 images
  four           : 20,216 images
  like           : 19,404 images
  mute           : 20,279 images
  ok             : 19,599 images
  one            : 19,910 images
  palm           : 19,828 images
  peace_inverted : 19,504 images
  peace          : 19,812 images
  rock           : 19,447 images
  stop_inverted  : 20,199 images
  stop           : 19,574 images
  three          : 19,610 images
  three2         : 19,452 images
  two_up_inverted: 19,765 images
  two_up         : 20,775 images

VALIDATION Split Class Distribution:
  call           : 5,638 images
  dislike        : 5,707 images
  fist           : 5,552 images
  four           : 5,776 images
  like           : 5,544 images
  mute           : 5,794 images
  ok             : 5,599 images
  one            : 5,688 images
  palm           : 5,665 images
  peace_inverted : 5,572 images
  peace          : 5,660 images
  rock           : 5,556 images
  stop_inverted  : 5,771 images
  stop           : 5,592 images
  three          : 5,603 images
  three2         : 5,557 images
  two_up_inverted: 5,647 images
  two_up         : 5,935 images

TEST Split Class Distribution:
  call           : 2,820 images
  dislike        : 2,855 images
  fist           : 2,778 images
  four           : 2,888 images
  like           : 2,773 images
  mute           : 2,898 images
  ok             : 2,801 images
  one            : 2,846 images
  palm           : 2,833 images
  peace_inverted : 2,788 images
  peace          : 2,831 images
  rock           : 2,779 images
  stop_inverted  : 2,887 images
  stop           : 2,797 images
  three          : 2,802 images
  three2         : 2,780 images
  two_up_inverted: 2,824 images
  two_up         : 2,969 images
Epoch 1/20 [Train]: 100%|███████████████████████████████████████| 6250/6250 [09:03<00:00, 11.49it/s, Loss=0.7352, Acc=71.30%]
Epoch 1/20 [Val]: 100%|█████████████████████████████████████████| 6366/6366 [05:24<00:00, 19.61it/s, Loss=0.7427, Acc=88.29%]
2025-08-18 11:59:24,388 - INFO - Best model saved with validation accuracy: 88.29%

Epoch 1/20:
  Train Loss: 0.7352, Train Acc: 71.30%
  Val Loss: 0.7427, Val Acc: 88.29%
  LR: 0.000020
  🚀 Great start! Model is learning
  *** NEW BEST MODEL: 88.29% ***
Epoch 2/20 [Train]: 100%|███████████████████████████████████████| 6250/6250 [09:03<00:00, 11.50it/s, Loss=0.4301, Acc=81.13%]
Epoch 2/20 [Val]: 100%|█████████████████████████████████████████| 6366/6366 [05:33<00:00, 19.11it/s, Loss=0.6479, Acc=91.33%]
2025-08-18 12:14:01,540 - INFO - Best model saved with validation accuracy: 91.33%

Epoch 2/20:
  Train Loss: 0.4301, Train Acc: 81.13%
  Val Loss: 0.6479, Val Acc: 91.33%
  LR: 0.000035
  🚀 Great start! Model is learning
  *** NEW BEST MODEL: 91.33% ***

Target accuracy 90.0% reached!

================================================================================
TRAINING COMPLETED!
================================================================================
Total training time: 29.1 minutes
Best validation accuracy: 91.33%
Target achieved: ✓

========================================
FINAL TEST SET EVALUATION
========================================
Epoch 0/20 [Val]: 100%|█████████████████████████████████████████| 3185/3185 [02:46<00:00, 19.10it/s, Loss=0.6505, Acc=91.22%]
2025-08-18 12:16:50,515 - INFO - Training completed. Results saved to model/checkpoints_fixed
Test Accuracy: 91.22%

Per-Class Test Accuracy:
  call           : 89.7%
  dislike        : 99.3%
  fist           : 96.4%
  four           : 91.1%
  like           : 93.9%
  mute           : 97.0%
  ok             : 91.5%
  one            : 82.8%
  palm           : 93.9%
  peace_inverted : 91.8%
  peace          : 83.3%
  rock           : 90.8%
  stop_inverted  : 92.1%
  stop           : 95.2%
  three          : 76.3%
  three2         : 88.4%
  two_up_inverted: 97.2%
  two_up         : 91.3%

TRAINING COMPLETED!
✓ Best validation accuracy: 91.33%
✓ Test accuracy: 91.22%
✓ Training time: 29.1 minutes
✓ Model parameters: 65,106,770
✓ Target achieved: Yes
✓ Ready for quantization: Yes

🎯 EXCELLENT!
```