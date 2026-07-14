# resnet-study
PyTorch implementation and study of ResNet for image classification.

### 01_baseline
{
    "model": "resnet34",
    "freeze_level": 63,
    "drop_out": 0.3,
    "epochs": 30,
    "batch_size": 128,
    "lr": 0.001,
    "optimizer": "sgd",
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "patience": 7,
    "label_smoothing": 0.1,
    "seed": 42,
    "save_dir": "study_results/01_baseline",
    "best_val_acc": 82.22,
    "test_acc": 81.77
}
![alt text](study_results/01_baseline/resnet34_f63_sgd_lr0.001_bs128/curves.png)
### 02_high_lr
{
    "model": "resnet34",
    "freeze_level": 63,
    "drop_out": 0.3,
    "epochs": 30,
    "batch_size": 128,
    "lr": 0.01,
    "optimizer": "sgd",
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "patience": 7,
    "label_smoothing": 0.1,
    "seed": 42,
    "save_dir": "study_results/02_high_lr",
    "best_val_acc": 85.46,
    "test_acc": 84.67
}
![alt text](study_results/02_high_lr/resnet34_f63_sgd_lr0.01_bs128/curves.png)
### 03_low_lr
{
    "model": "resnet34",
    "freeze_level": 63,
    "drop_out": 0.3,
    "epochs": 30,
    "batch_size": 128,
    "lr": 0.0001,
    "optimizer": "sgd",
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "patience": 7,
    "label_smoothing": 0.1,
    "seed": 42,
    "save_dir": "study_results/03_low_lr",
    "best_val_acc": 70.48,
    "test_acc": 69.83
}
![alt text](study_results/03_low_lr/resnet34_f63_sgd_lr0.0001_bs128/curves.png)
### 04_freeze_fc_only
{
    "model": "resnet34",
    "freeze_level": 1,
    "drop_out": 0.3,
    "epochs": 30,
    "batch_size": 128,
    "lr": 0.001,
    "optimizer": "sgd",
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "patience": 7,
    "label_smoothing": 0.1,
    "seed": 42,
    "save_dir": "study_results/04_freeze_fc_only",
    "best_val_acc": 60.44,
    "test_acc": 59.57
}
![alt text](study_results/04_freeze_fc_only/resnet34_f1_sgd_lr0.001_bs128/curves.png)
### 05_freeze_partial
{
    "model": "resnet34",
    "freeze_level": 7,
    "drop_out": 0.3,
    "epochs": 30,
    "batch_size": 128,
    "lr": 0.001,
    "optimizer": "sgd",
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "patience": 7,
    "label_smoothing": 0.1,
    "seed": 42,
    "save_dir": "study_results/05_freeze_partial",
    "best_val_acc": 79.06,
    "test_acc": 79.13
}
![alt text](study_results/05_freeze_partial/resnet34_f7_sgd_lr0.001_bs128/curves.png)
### 06_optimizer_adam
{
    "model": "resnet34",
    "freeze_level": 63,
    "drop_out": 0.3,
    "epochs": 30,
    "batch_size": 128,
    "lr": 0.0001,
    "optimizer": "adam",
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "patience": 7,
    "label_smoothing": 0.1,
    "seed": 42,
    "save_dir": "study_results/06_optimizer_adam",
    "best_val_acc": 85.02,
    "test_acc": 84.09
}
![alt text](study_results/06_optimizer_adam/resnet34_f63_adam_lr0.0001_bs128/curves.png)