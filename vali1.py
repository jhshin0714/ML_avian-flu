from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import sys
import os

def evaluate_model(model_path, test_data_dir, batch_size=16):
    # 모델 로드
    model = load_model(model_path)
    
    # 모델의 입력 크기를 확인
    input_shape = model.input_shape[1:3]

    # 데이터 전처리
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=input_shape,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False)

    # 예측 수행
    test_steps = test_generator.samples // test_generator.batch_size
    test_generator.reset()
    pred = model.predict(test_generator, steps=test_steps, verbose=1)
    pred_classes = np.round(pred).astype(int)

    # 실제 라벨 가져오기
    true_classes = test_generator.classes[:len(pred_classes)]

    # 점수 계산
    precision = precision_score(true_classes, pred_classes)
    recall = recall_score(true_classes, pred_classes)
    f1 = f1_score(true_classes, pred_classes)

    return precision, recall, f1

if __name__ == "__main__":
    model_path = sys.argv[1]
    test_data_dir = './data/test'

    precision, recall, f1 = evaluate_model(model_path, test_data_dir)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

