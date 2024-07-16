# 필요한 라이브러리를 임포트합니다.
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Iris 데이터셋을 로드합니다.
iris = load_iris()
X = iris.data  # 특징 데이터
y = iris.target  # 라벨 데이터

# 데이터셋을 학습 세트와 테스트 세트로 나눕니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Decision Tree Classifier 모델을 초기화하고 학습시킵니다.
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 테스트 세트로 예측을 수행합니다.
y_pred = clf.predict(X_test)

# 정확도와 분류 보고서를 출력합니다.
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)

