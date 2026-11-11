"""CIFAR-10 소프트맥스 선형 분류기의 RGB와 RGB+소벨 입력을 비교한다."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from torchvision.datasets import CIFAR10


SEED = 42
NUM_CLASSES = 10
EPOCHS = 40
BATCH_SIZE = 128
LEARNING_RATE = 0.03
L2 = 0.001


def sobel_magnitude(gray):
    padded = np.pad(gray, ((0, 0), (1, 1), (1, 1)), mode="reflect")
    tl, t, tr = padded[:, :-2, :-2], padded[:, :-2, 1:-1], padded[:, :-2, 2:]
    left, right = padded[:, 1:-1, :-2], padded[:, 1:-1, 2:]
    bl, b, br = padded[:, 2:, :-2], padded[:, 2:, 1:-1], padded[:, 2:, 2:]
    gx = (tr - tl) + 2 * (right - left) + (br - bl)
    gy = (bl - tl) + 2 * (b - t) + (br - tr)
    return np.hypot(gx, gy) / np.float32(4 * np.sqrt(2))


def load_data(root, train, download, use_sobel):
    dataset = CIFAR10(root=str(root), train=train, download=download)
    y = np.asarray(dataset.targets)
    rgb = dataset.data.reshape(len(y), -1).astype(np.float32, order="C")
    rgb /= 255.0
    if not use_sobel:
        return rgb, y, dataset.classes

    rgb = rgb.reshape(-1, 32, 32, 3)
    features = np.empty((*rgb.shape[:-1], 4), dtype=np.float32)
    features[..., :3] = rgb
    for start in range(0, len(rgb), 512):
        batch = rgb[start:start + 512]
        gray = 0.299 * batch[..., 0] + 0.587 * batch[..., 1] + 0.114 * batch[..., 2]
        features[start:start + len(batch), ..., 3] = sobel_magnitude(gray)
    return features.reshape(len(y), -1), y, dataset.classes


class SoftmaxLinearClassifier:
    def fit(self, X, y):
        # 평균은 각 fold의 학습 데이터에서만 계산한다.
        self.pixel_mean = X.mean(axis=0)
        X_centered = X - self.pixel_mean
        self.W = np.zeros((X.shape[1], NUM_CLASSES), dtype=np.float32)
        self.b = np.zeros(NUM_CLASSES, dtype=np.float32)
        rng = np.random.default_rng(SEED)

        for _ in range(EPOCHS):
            order = rng.permutation(len(y))
            for start in range(0, len(y), BATCH_SIZE):
                indices = order[start:start + BATCH_SIZE]
                X_batch, y_batch = X_centered[indices], y[indices]
                scores = X_batch @ self.W + self.b
                scores -= scores.max(axis=1, keepdims=True)
                probabilities = np.exp(scores)
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                probabilities[np.arange(len(indices)), y_batch] -= 1.0
                grad_scores = probabilities / len(indices)
                grad_W = X_batch.T @ grad_scores + L2 * self.W
                grad_b = grad_scores.sum(axis=0)
                self.W -= LEARNING_RATE * grad_W
                self.b -= LEARNING_RATE * grad_b
        return self

    def predict(self, X):
        return ((X - self.pixel_mean) @ self.W + self.b).argmax(axis=1)


def evaluate(y, predictions):
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predictions, labels=np.arange(NUM_CLASSES), zero_division=0
    )
    aggregate = {
        "accuracy": float(np.mean(y == predictions)),
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
    }
    classes = [
        {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(n)}
        for p, r, f, n in zip(precision, recall, f1, support)
    ]
    return aggregate, classes


def run_model(root, download, use_sobel):
    X_train, y_train, labels = load_data(root, train=True, download=download, use_sobel=use_sobel)
    name = "RGB+Sobel" if use_sobel else "RGB"
    print(f"{name} | 학습 {X_train.shape}", flush=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
        model = SoftmaxLinearClassifier().fit(X_train[train_idx], y_train[train_idx])
        metrics, _ = evaluate(y_train[val_idx], model.predict(X_train[val_idx]))
        folds.append({"fold": fold, **metrics})
        print(f"{name} | fold {fold}/5 | 정확도 {metrics['accuracy']:.2%}", flush=True)

    model = SoftmaxLinearClassifier().fit(X_train, y_train)
    feature_count = X_train.shape[1]
    train_size = len(y_train)
    del X_train, y_train
    X_test, y_test, test_labels = load_data(root, train=False, download=download, use_sobel=use_sobel)
    if test_labels != labels:
        raise ValueError("학습과 테스트의 클래스 순서가 다릅니다.")
    predictions = model.predict(X_test)
    metrics, classes = evaluate(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, labels=np.arange(NUM_CLASSES))
    print(f"{name} | 테스트 {len(y_test):,}장 | 정확도 {metrics['accuracy']:.2%}", flush=True)
    return {
        "features": feature_count,
        "train_size": train_size,
        "folds": folds,
        "cv_mean": {key: float(np.mean([row[key] for row in folds])) for key in metrics},
        "test": metrics,
        "classes": classes,
        "confusion_matrix": matrix.tolist(),
    }, labels, y_test, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--download", action="store_true", help="CIFAR-10이 없으면 내려받기")
    args = parser.parse_args()
    with threadpool_limits(limits=2, user_api="blas"):
        rgb, labels, y_test, rgb_predictions = run_model(args.data_dir, args.download, use_sobel=False)
        sobel, sobel_labels, sobel_y, sobel_predictions = run_model(
            args.data_dir, args.download, use_sobel=True
        )
    if labels != sobel_labels or not np.array_equal(y_test, sobel_y):
        raise ValueError("두 모델의 테스트 데이터가 다릅니다.")
    paired = confusion_matrix(y_test == rgb_predictions, y_test == sobel_predictions, labels=[False, True])
    result = {
        "config": {"l2": L2, "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                   "learning_rate": LEARNING_RATE, "folds": 5, "test_size": len(y_test)},
        "labels": labels,
        "models": {"RGB": rgb, "RGB+Sobel": sobel},
        "paired_correctness": paired.tolist(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
