import os
import random
import pickle
import numpy as np
import kagglehub
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset

<<<<<<< Updated upstream
# %%
# 난수 생성의 기준값(Seed) 설정
# 같은 Seed를 사용하면 매번 동일한 난수가 생성되어 실험을 재현할 수 있음
SEED = 42
=======
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import argparse
import logging
import json
import csv
from datetime import datetime
>>>>>>> Stashed changes

# ==========================================
# 1. 하이퍼파라미터 및 설정 (Parser 확장)
# ==========================================
def get_args():
    parser = argparse.ArgumentParser(description="Advanced CIFAR-100 Training")
    
    # 모델 관련
    parser.add_argument("--model", type=str, default="resnet34", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--freeze_level", type=int, default=63, 
                        help="Binary flags for training: bit0:FC, bit1:L4, bit2:L3, bit3:L2, bit4:L1, bit5:Stem (e.g., 3=FC+L4)")
    parser.add_argument("--drop_out", type=float, default=0.3)
    
    # 학습 관련
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    
    # 시스템 및 재현성
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="results")
    
    try:
        return parser.parse_args()
    except:
        return parser.parse_args([])

args = get_args()

# 폴더 생성 및 로그 설정
exp_name = f"{args.model}_f{args.freeze_level}_{args.optimizer}_lr{args.lr}_bs{args.batch_size}"
current_save_dir = os.path.join(args.save_dir, exp_name)
os.makedirs(current_save_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(current_save_dir, "train.log"),
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)

# ==========================================
# 2. 재현성 설정
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 3. 모델 구축 및 2진수 기반 동결 로직
# ==========================================
def build_model(args):
    # 모델 동적 로드
    model_name = args.model.lower()
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # FC 레이어 수정
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(args.drop_out),
        nn.Linear(num_ftrs, 100)
    )
    
    # 2진수 기반 레이어 동결 (freeze_level)
    # Bit 0: fc, 1: layer4, 2: layer3, 3: layer2, 4: layer1, 5: stem(conv1, bn1)
    mapping = {
        0: [model.fc],
        1: [model.layer4],
        2: [model.layer3],
        3: [model.layer2],
        4: [model.layer1],
        5: [model.conv1, model.bn1, model.maxpool]
    }
    
    print("\n--- Layer Trainable Status (Binary Check) ---")
    for bit, modules in mapping.items():
        is_trainable = (args.freeze_level >> bit) & 1
        for module in modules:
            for param in module.parameters():
                param.requires_grad = bool(is_trainable)
        
        status = "Trainable" if is_trainable else "Frozen"
        print(f"Bit {bit} ({list(mapping.keys())[bit]}): {status}")
    
    return model.to(device)

model = build_model(args)

# ==========================================
# 4. 데이터 준비 (CIFAR-100)
# ==========================================
path = kagglehub.dataset_download("fedesoriano/cifar100")

class CustomCIFAR100(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        filename = "train" if train else "test"
        with open(os.path.join(root, filename), "rb") as f:
            entry = pickle.load(f, encoding="latin1")
        self.data = entry["data"].reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
        self.labels = entry["fine_labels"]
        with open(os.path.join(root, "meta"), "rb") as f:
            meta = pickle.load(f, encoding="latin1")
        self.classes = meta["fine_label_names"]

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        img, label = Image.fromarray(self.data[idx]), self.labels[idx]
        if self.transform: img = self.transform(img)
        return img, label

MEAN, STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
train_trf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])
test_trf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

full_train = CustomCIFAR100(path, train=True, transform=train_trf)
val_ds = CustomCIFAR100(path, train=True, transform=test_trf)
test_ds = CustomCIFAR100(path, train=False, transform=test_trf)

indices = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(args.seed))
train_idx, val_idx = indices[:45000], indices[45000:]
train_loader = DataLoader(Subset(full_train, train_idx), batch_size=args.batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=args.batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

# ==========================================
# 5. 최적화 및 학습 루프
# ==========================================
criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
trainable_params = [p for p in model.parameters() if p.requires_grad]

if args.optimizer == "sgd":
    optimizer = optim.SGD(trainable_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
else:
    optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

<<<<<<< Updated upstream
# %%
train_size = int(len(train_dataset) * 0.9)
val_size = len(train_dataset) - train_size

generator = torch.Generator().manual_seed(SEED)

indices = torch.randperm(len(train_dataset), generator=generator)

train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_data = Subset(train_dataset, train_indices)
val_data = Subset(val_dataset, val_indices)

# %%
train_loader = DataLoader(
    train_data,
    batch_size=128,
    shuffle=True,
    pin_memory=True
)

val_loader = DataLoader(
    val_data,
    batch_size=128,
    shuffle=False,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    # num_workers=4,        # 데이터를 읽는 Worker(Process) 개수
    pin_memory=True         # CPU 메모리를 Pinned Memory(Page-Locked Memory)로 할당, images = images.to(device)를 실행할 때 CPU → GPU 전송 속도를 높여 줍니다
)

# %%
print("Train Dataset :", len(train_dataset))
print("Validation :", len(val_data))
print("Test Dataset :", len(test_dataset))
print("Number of Classes :", len(train_dataset.classes))

images, labels = next(iter(train_loader))

print(images.shape)
print(labels.shape)

# %%
weights = models.ResNet34_Weights.IMAGENET1K_V1

resnet = models.resnet34(weights=weights)

print("Original FC")
print(resnet.fc)

num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 100)

resnet = resnet.to(device)

print("\nModified FC")
print(resnet.fc)

# %%
# # layer1, layer2는 ImageNet 특징 유지
# for param in resnet.layer1.parameters():
#     param.requires_grad = False

# for param in resnet.layer2.parameters():
#     param.requires_grad = False

# # layer3부터 CIFAR에 적응
# for param in resnet.layer3.parameters():
#     param.requires_grad = True

# for param in resnet.layer4.parameters():
#     param.requires_grad = True

# for param in resnet.fc.parameters():
#     param.requires_grad = True

for param in resnet.parameters():
    param.requires_grad = True

# %%
criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1  # 정답 클래스의 확률을 0.9정도로 낮추고, 나머지 클래스에 0.1을 나누어 분배하여 과적합 줄임
)

optimizer = torch.optim.SGD(
    [
        { "params": resnet.parameters(), "lr": 1e-2 },
    ]
    , momentum=0.9      # 이전 기울기를 일부 반영하여 더 빠르고 안정적으로 최적점을 찾음
    , weight_decay=1e-3 # 큰 가중치가 생기는 것을 방지하여 과적합(Overfitting)을 줄임
)        

# %%
# Learning Rate Scheduler
epochs = 40

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs
)

# Mixed Precision
=======
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
>>>>>>> Stashed changes
scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

best_val_acc = 0.0
counter = 0
history = {"t_loss": [], "v_loss": [], "t_acc": [], "v_acc": []}

<<<<<<< Updated upstream
# %%
# History
train_losses = []
val_losses = []

train_accs = []
val_accs = []

# %%
# Model Information
print(resnet)

print()

trainable = sum(
    p.numel()
    for p in resnet.parameters()
    if p.requires_grad
)

total = sum(
    p.numel()
    for p in resnet.parameters()
)

print(f"Trainable Parameters : {trainable:,}")
print(f"Total Parameters     : {total:,}")

# %%
# epochs 횟수만큼 전체 학습 반복
for epoch in range(epochs):

    # ============================
    # Train Mode
    # ============================
    resnet.train()
    # 모델을 학습 모드로 변경
    # Dropout, BatchNorm 등이 학습 방식으로 동작

    train_loss = 0.0      # Epoch 동안의 총 Train Loss
    train_correct = 0     # 맞춘 이미지 개수
    train_total = 0       # 전체 이미지 개수

    # train_loader에서 Batch 단위로 데이터 가져오기
    for images, labels in train_loader:

        # 이미지를 GPU(CUDA) 또는 CPU로 이동
        # non_blocking=True는 pin_memory=True와 함께 사용하면
        # CPU → GPU 전송을 비동기로 수행하여 속도를 향상시킴
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 이전 Batch에서 계산된 Gradient 초기화
=======
for epoch in range(args.epochs):
    model.train()
    t_loss, t_corr, t_total = 0, 0, 0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
>>>>>>> Stashed changes
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
            out = model(imgs)
            loss = criterion(out, lbls)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        t_loss += loss.item()
        t_corr += out.max(1)[1].eq(lbls).sum().item()
        t_total += lbls.size(0)
    
    model.eval()
    v_loss, v_corr, v_total = 0, 0, 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            v_loss += criterion(out, lbls).item()
            v_corr += out.max(1)[1].eq(lbls).sum().item()
            v_total += lbls.size(0)

    # 에폭 결과 정리
    train_acc, val_acc = 100.*t_corr/t_total, 100.*v_corr/v_total
    history["t_loss"].append(t_loss/len(train_loader)); history["v_loss"].append(v_loss/len(val_loader))
    history["t_acc"].append(train_acc); history["v_acc"].append(val_acc)
    
    msg = f"Ep {epoch+1:02d} | TLoss:{history['t_loss'][-1]:.3f} TAcc:{train_acc:.2f}% | VLoss:{history['v_loss'][-1]:.3f} VAcc:{val_acc:.2f}%"
    print(msg); logging.info(msg)
    scheduler.step()

    # 모델 저장 및 조기 종료
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(current_save_dir, "best_model.pth"))
        counter = 0
    else:
        counter += 1
        if counter >= args.patience:
            print("Early Stopping..."); break

# ==========================================
# 6. 최종 평가 및 시각화 저장
# ==========================================
def save_plots(history, save_dir):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(history['t_loss'], label='Train'); plt.plot(history['v_loss'], label='Val')
    plt.title('Loss'); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(history['t_acc'], label='Train'); plt.plot(history['v_acc'], label='Val')
    plt.title('Accuracy'); plt.legend()
    plt.savefig(os.path.join(save_dir, "curves.png"))
    plt.close()

save_plots(history, current_save_dir)

# 테스트 세트 검증
model.load_state_dict(torch.load(os.path.join(current_save_dir, "best_model.pth")))
model.eval()
all_lbl, all_prd = [], []
with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs = imgs.to(device)
        out = model(imgs)
        all_lbl.extend(lbls.numpy())
        all_prd.extend(out.max(1)[1].cpu().numpy())

test_acc = accuracy_score(all_lbl, all_prd) * 100
print(f"\nFinal Test Accuracy: {test_acc:.2f}%")

<<<<<<< Updated upstream
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

test_loss /= len(test_loader)
test_acc = 100 * correct / total

print("=" * 60)
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_acc:.2f}%")
print("=" * 60)

# %%
# Loss Curve
plt.figure(figsize=(8,5))

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Loss Curve")

plt.grid(True)

plt.legend()

plt.show()


# %%
# Accuracy Curve
plt.figure(figsize=(8,5))

plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.title("Accuracy Curve")

plt.grid(True)

plt.legend()

plt.show()

# %%
# Prediction Visualization
classes = train_dataset.classes

images, labels = next(iter(test_loader))

images_gpu = images.to(device)

resnet.eval()

with torch.no_grad():

    outputs = resnet(images_gpu)

    _, predicted = torch.max(outputs, 1)

predicted = predicted.cpu()

# Normalize 복원
mean = torch.tensor(MEAN).view(3,1,1)
std = torch.tensor(STD).view(3,1,1)

plt.figure(figsize=(15,8))

for i in range(8):

    img = images[i] * std + mean
    img = img.clamp(0,1)

    plt.subplot(2,4,i+1)

    plt.imshow(img.permute(1,2,0))

    color = "green" if predicted[i] == labels[i] else "red"

    plt.title(
        f"GT : {classes[labels[i]]}\n"
        f"Pred : {classes[predicted[i]]}",
        fontsize=9,
        color=color
    )

    plt.axis("off")

plt.tight_layout()

plt.show()

# %%
print("\nTraining Finished.")
=======
# 리포트 및 요약 저장
with open(os.path.join(current_save_dir, "summary.json"), "w") as f:
    json.dump({**vars(args), "best_val_acc": best_val_acc, "test_acc": test_acc}, f, indent=4)
>>>>>>> Stashed changes

with open(os.path.join(current_save_dir, "classification_report.txt"), "w") as f:
    f.write(classification_report(all_lbl, all_prd, target_names=full_train.classes))

print(f"Completed. Results in: {current_save_dir}")