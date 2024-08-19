# -*- coding: utf-8 -*-

# 패키지 가져오기
import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
import re

def load_data(filepath):
    # 데이터셋 로드
    data = pd.read_csv(filepath, skipinitialspace=True)
    # 'Unnamed:' 칼럼 제거
    data = data.loc[:, ~data.columns.str.startswith('Unnamed: ')]
    # 칼럼 이름 변경하기
    data.columns = [re.sub('[^A-Za-z_%#\(\)\+\-\.\?\!\<\>\[\]\=/가-힣ㄱ-ㅎㅏ-ㅣ0-9]', '_', col.strip()) for col in data.columns]

    return data

def preprocess_data(data, target_col):
    # X와 y 분리
    X = data.drop(target_col, axis=1)
    y = data[target_col].values

    # 훈련용-시험용 데이터셋으로 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 칼럼 유형 분류
    cat_columns = X_train.select_dtypes(include='object').columns
    num_columns = X_train.select_dtypes(exclude='object').columns

    # 결측값 바꾸기
    X_train[cat_columns] = X_train[cat_columns].fillna("<NA>")
    X_test[cat_columns] = X_test[cat_columns].fillna("<NA>")
    medians = X_train[num_columns].median()
    X_train[num_columns] = X_train[num_columns].fillna(medians)
    X_test[num_columns] = X_test[num_columns].fillna(medians)

    # 범주형 칼럼 인코딩
    encoder = preprocessing.OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-1)
    train_cat_values = encoder.fit_transform(X_train[cat_columns])
    test_cat_values = encoder.transform(X_test[cat_columns])

    # 수치 칼럼 스케일링
    scaler = preprocessing.StandardScaler()
    train_num_values = scaler.fit_transform(X_train[num_columns])
    test_num_values = scaler.transform(X_test[num_columns])

    # 훈련용과 시험용 데이터셋 재구성
    X_train = np.hstack((train_cat_values, train_num_values))
    X_test = np.hstack((test_cat_values, test_num_values))

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    # 가장 적당한 매개변수로 모델 구축
    model = Ridge(**{
        "alpha": 2.309,
        "solver": "sparse_cg",
        "max_iter": 673,
        "tol": 0.0
    })

    # 구축된 모델 훈련
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):
    # 시험용 데이터셋을 사용한 예측
    y_pred = model.predict(X_test)
    y_yhat = pd.DataFrame({'Real': y_test, 'Pred': y_pred})
    print(y_yhat.reset_index(drop=True).round(3), '\n')

    # 성능 지표 계산
    print("Mean Absolute Error")
    print(mean_absolute_error(y_test, y_pred), '\n')

def main():
    # 데이터 로드와 탐색
    data = load_data("낙찰가_dataset.csv")
    print('Dataset Shape:', data.shape, '\n')
    print(data.head().round(3), '\n')
    data.info()
    print()
    print(data.describe(), '\n')

    # 데이터 전처리
    X_train, X_test, y_train, y_test = preprocess_data(data, '투찰금액(원)')

    # 모델 훈련과 평가
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
