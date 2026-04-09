import os
import glob
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import numpy as np
from sklearn.metrics import f1_score

# =========================================================
# 1. 커스텀 Dataset 정의
# =========================================================
class BicycleRoadDataset(Dataset):
    def __init__(self, data_list, transform=None):
        """
        data_list: list of dict -> [{'img_path': str, 'labels': [d, o, u]}, ...]
        """
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        img_path = item['img_path']
        # labels: [is_defect, is_obstacle, is_unpaved]
        labels = torch.tensor(item['labels'], dtype=torch.float32)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # 이미지 손상 시 빈 검은색 이미지 반환 (예외 처리)
            image = Image.new("RGB", (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        return image, labels

# =========================================================
# 2. 데이터 파싱 함수 (JSON -> Data List)
# =========================================================
def parse_ai_hub_data(base_path, max_samples=None):
    """
    ai-hub 자전거도로 주행 데이터에서 이미지와 라벨(JSON) 매핑
    """
    print("라벨링 데이터(JSON) 검색 중...")
    
    # 윈도우 환경 대응 및 범용적인 탐색을 위해 glob 사용
    label_search_path = os.path.join(base_path, "**", "02.라벨링데이터", "**", "*.json")
    json_files = glob.glob(label_search_path, recursive=True)
    
    # 만일 폴더 구조가 달라 못찾는 경우 대비 (추출된 임시 경로 등)
    if not json_files:
        json_files = glob.glob(os.path.join(base_path, "**", "*.json"), recursive=True)
        
    print(f"총 {len(json_files)}개의 JSON 파일을 찾았습니다.")
    
    data_list = []
    
    for j_path in tqdm(json_files, desc="JSON 파싱 및 라벨 추출"):
        with open(j_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except:
                continue
                
        # "images" -> "file_name" 필드 추출
        img_info = data.get("images", {})
        file_name = img_info.get("file_name", "")
        if not file_name:
            continue
            
        # 해당 이미지 경로 추정 (JSON 경로에서 '02.라벨링데이터' -> '01.원천데이터'로 치환)
        # 상황에 따라 동일 폴더 내 확장자만 다를 수도 있음
        img_path_1 = j_path.replace("02.라벨링데이터", "01.원천데이터").replace(".json", ".jpg")
        img_path_2 = j_path.replace(".json", ".jpg")  # 압축 푼 폴더에 같이 있을 경우
        
        target_img_path = None
        if os.path.exists(img_path_1):
            target_img_path = img_path_1
        elif os.path.exists(img_path_2):
            target_img_path = img_path_2
        else:
            continue # 이미지를 찾을 수 없음
            
        # 클래스 플래그
        is_defect = 0
        is_obstacle = 0
        is_unpaved = 0
        
        # 어노테이션에서 키워드 추출
        anns = data.get("annotations", [])
        for ann in anns:
            cat = ann.get("category_name", "")
            sub = ann.get("sub_category_name", "")
            combo = f"{cat} {sub}".strip()
            
            # 위험 요소 키워드 매칭
            if "크랙" in combo or "파손" in combo or "단차" in combo or "도로결함" in combo:
                is_defect = 1
            if "불법주정차" in combo or "보행자" in combo:
                is_obstacle = 1
            if "비포장" in combo:
                is_unpaved = 1
                
        data_list.append({
            "img_path": target_img_path,
            "labels": [is_defect, is_obstacle, is_unpaved]
        })
        
        if max_samples and len(data_list) >= max_samples:
            break
            
    print(f"유효한 쌍 (이미지-라벨) 총 {len(data_list)}개 확보")
    return data_list

# =========================================================
# 3. 모델 및 학습 메인 파이프라인
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"data\ai-hub\187.자전거도로_주행_데이터")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_samples", type=int, default=None, help="테스트용으로 샘플 수 제한")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 데이터 로드
    data_list = parse_ai_hub_data(args.data_dir, args.max_samples)
    if len(data_list) < 10:
        print("데이터가 부족합니다. 경로를 확인해주세요.")
        return

    # Train / Val / Test Split (70% / 15% / 15%)
    train_val_list, test_list = train_test_split(data_list, test_size=0.15, random_state=42)
    train_list, val_list = train_test_split(train_val_list, test_size=0.15 / 0.85, random_state=42)
    
    print(f"DataSet Split -> Train: {len(train_list)}, Val: {len(val_list)}, Test: {len(test_list)}")

    # Transforms (ImageNet 표준)
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = BicycleRoadDataset(train_list, transform_train)
    val_ds = BicycleRoadDataset(val_list, transform_test)
    test_ds = BicycleRoadDataset(test_list, transform_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 모델 정의: EfficientNet-B0
    # 출력 클래스 3개 (is_defect, is_obstacle, is_unpaved)
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 3)
    model = model.to(device)

    # 손실함수 및 옵티마이저 (Multi-label 이므로 BCE 로그로짓 사용)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    save_path = "models/dl/road_image_efficientnet.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Training Loop
    print("\n--- Training Start ---")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            pbar.set_postfix({'loss': loss.item()})
            
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                preds = torch.sigmoid(outputs) > 0.5
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                
        val_loss = val_loss / len(val_loader.dataset)
        
        # Calculate Validation F1 Score (Macro)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        print(f"Epoch {epoch+1} -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f">>> Best 모델 저장 완료 (Val Loss: {best_val_loss:.4f})")

    # =========================================================
    # 4. 모델 테스트 및 검증
    # =========================================================
    print("\n--- Test Evaluation ---")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    test_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            
            preds = torch.sigmoid(outputs) > 0.5
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    test_loss = test_loss / len(test_loader.dataset)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    print("\n[ 테스트 결과 요약 ]")
    print(f"Test Loss: {test_loss:.4f}")
    classes = ["도로결함(is_defect)", "장애물(is_obstacle)", "비포장(is_unpaved)"]
    
    for i, cls_name in enumerate(classes):
        f1 = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        print(f" - {cls_name} F1 Score: {f1:.4f}")

    print("\n이후 실제 이미지 서비스 변환 예시:")
    print(" 1) img = Image.open('test_street.jpg')")
    print(" 2) img_tensor = transform_test(img).unsqueeze(0).to(device)")
    print(" 3) probs = torch.sigmoid(model(img_tensor))[0].cpu().numpy()")
    print(" 4) print(f'결함확률 {probs[0]:.2f}, 장애물확률 {probs[1]:.2f}, 비포장확률 {probs[2]:.2f}')")
    
if __name__ == '__main__':
    main()
