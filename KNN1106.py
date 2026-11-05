from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from torchvision.datasets import CIFAR10
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

DATA_DIR = Path(__file__).resolve().parent / "data"

train_data = CIFAR10(root=str(DATA_DIR), train=True, download=True)
test_data = CIFAR10(root=str(DATA_DIR), train=False, download=True)


X_train = train_data.data.reshape(-1, 3072).astype(np.float32)
X_train /= 255.0
y_train = np.asarray(train_data.targets)

X_test = test_data.data.reshape(-1, 3072).astype(np.float32)
X_test /= 255.0
y_test = np.asarray(test_data.targets)

X_train, _, y_train, _ = train_test_split(
    X_train,
    y_train,
    train_size=5000,
    random_state=42,
    stratify=y_train,
)

X_test, _, y_test, _ = train_test_split(
    X_test,
    y_test,
    train_size=1000,
    random_state=42,
    stratify=y_test,
)




print("전체 학습 데이터:", X_train.shape)
print("전체 테스트 데이터:", X_test.shape)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

class MyKNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _find_neighbors_one(self, x, max_k):
        distances = [
            self._euclidean_distance(x, x_train)
            for x_train in self.X_train
        ]
        nearest_indices = np.argsort(distances)[:max_k]

        return [self.y_train[i] for i in nearest_indices]
    def find_neighbors(self, X_test, max_k=5):
        return np.array([
            self._find_neighbors_one(x, max_k)
            for x in X_test
        ])

    def predict(self, neighbor_labels):
        return np.array([
            Counter(labels[:self.k]).most_common(1)[0][0]
            for labels in neighbor_labels
        ])


class MyKNN_man:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def _manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))

    def _find_neighbors_one(self, x, max_k):
        distances = [
            self._manhattan_distance(x, x_train)
            for x_train in self.X_train
        ]

        nearest_indices = np.argsort(distances)[:max_k]

        return [self.y_train[i] for i in nearest_indices]

    def find_neighbors(self, X_test, max_k=5):
        return np.array([
            self._find_neighbors_one(x, max_k)
            for x in X_test
        ])

    def predict(self, neighbor_labels):
        return np.array([
            Counter(labels[:self.k]).most_common(1)[0][0]
            for labels in neighbor_labels
        ])



ks = [1, 3, 5]

model_classes = {
    "euclidean": MyKNN,
    "manhattan": MyKNN_man,
}


records = []



for fold, (train_idx, val_idx) in enumerate(
    cv.split(X_train, y_train),
    start=1,
):

    X_fold_train = X_train[train_idx]
    y_fold_train = y_train[train_idx]


    X_val = X_train[val_idx]
    y_val = y_train[val_idx]

    print(f"\n===== Fold {fold}/5 =====", flush=True)
    print("학습:", X_fold_train.shape)
    print("검증:", X_val.shape)

    for metric, model_class in model_classes.items():
        print(f"{metric} 거리 계산 시작...", flush=True)


        model = model_class()

  
        model.fit(X_fold_train, y_fold_train)


        neighbor_labels = model.find_neighbors(
            X_val,
            max_k=max(ks),
        )

        for k in ks:
            model.k = k
            y_val_pred = model.predict(neighbor_labels)

            accuracy = accuracy_score(y_val, y_val_pred)

            precision = precision_score(
                y_val,
                y_val_pred,
                average="macro",
                zero_division=0,
            )

            recall = recall_score(
                y_val,
                y_val_pred,
                average="macro",
                zero_division=0,
            )

            f1 = f1_score(
                y_val,
                y_val_pred,
                average="macro",
                zero_division=0,
            )

            records.append({
                "fold": fold,
                "metric": metric,
                "k": k,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            })

            print(
                f"{metric}, k={k} | "
                f"Accuracy={accuracy:.4f}, "
                f"Precision={precision:.4f}, "
                f"Recall={recall:.4f}, "
                f"F1={f1:.4f}",
                flush=True,
            )

    del model, neighbor_labels
    del X_fold_train, y_fold_train, X_val, y_val



results_df = pd.DataFrame(records)

summary = results_df.groupby(
    ["metric", "k"]
).agg(
    accuracy_mean=("accuracy", "mean"),
    accuracy_std=("accuracy", "std"),
    precision_mean=("precision", "mean"),
    recall_mean=("recall", "mean"),
    f1_mean=("f1", "mean"),
)

print("\n===== 5-fold 평균 검증 결과 =====")
print(summary.round(4).to_string())


best_metric, best_k = summary["accuracy_mean"].idxmax()
best_k = int(best_k)

print(f"\n선택한 거리 방식: {best_metric}")
print(f"선택한 k: {best_k}")
print(
    "평균 검증 정확도:",
    f"{summary.loc[(best_metric, best_k), 'accuracy_mean']:.4f}",
)


final_model = model_classes[best_metric](k=best_k)


final_model.fit(X_train, y_train)

print("\n최종 테스트 거리 계산 시작...", flush=True)


test_neighbor_labels = final_model.find_neighbors(
    X_test,
    max_k=best_k,
)

y_test_pred = final_model.predict(test_neighbor_labels)

test_accuracy = accuracy_score(y_test, y_test_pred)

test_precision = precision_score(
    y_test,
    y_test_pred,
    average="macro",
    zero_division=0,
)

test_recall = recall_score(
    y_test,
    y_test_pred,
    average="macro",
    zero_division=0,
)

test_f1 = f1_score(
    y_test,
    y_test_pred,
    average="macro",
    zero_division=0,
)

print("\n===== 최종 테스트 결과 =====")
print(f"거리 방식: {best_metric}")
print(f"k: {best_k}")
print(f"Accuracy:  {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1-score:  {test_f1:.4f}")
    