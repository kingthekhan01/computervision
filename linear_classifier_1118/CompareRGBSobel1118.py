from pathlib import Path
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from threadpoolctl import threadpool_limits
from torchvision.datasets import CIFAR10

from LinearClassifier1118 import DATA_DIR, SEED, WEIGHT_SCALE, MyLinearClassifier, markdown_table


OUTPUT_DIR = Path(__file__).resolve().parent / "rgb_sobel_comparison"


def sobel_filter(gray):
    padded = np.pad(gray, ((0, 0), (1, 1), (1, 1)), mode="reflect")
    top_left = padded[:, :-2, :-2]
    top = padded[:, :-2, 1:-1]
    top_right = padded[:, :-2, 2:]
    left = padded[:, 1:-1, :-2]
    right = padded[:, 1:-1, 2:]
    bottom_left = padded[:, 2:, :-2]
    bottom = padded[:, 2:, 1:-1]
    bottom_right = padded[:, 2:, 2:]
    gx = (top_right - top_left) + 2 * (right - left) + (bottom_right - bottom_left)
    gy = (bottom_left - top_left) + 2 * (bottom - top) + (bottom_right - top_right)
    return np.hypot(gx, gy) / np.float32(4 * np.sqrt(2))


def make_features(rgb):
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    edge = sobel_filter(gray)
    return np.concatenate([rgb, edge[..., None]], axis=-1)


def make_models():
    rgb = MyLinearClassifier()
    rng = np.random.default_rng(SEED)
    np.testing.assert_array_equal(
        rgb.W, rng.normal(0.0, WEIGHT_SCALE, (10, 3072))
    )
    weights = np.empty((10, 32, 32, 4))
    weights[..., :3] = rgb.W.reshape(10, 32, 32, 3)
    weights[..., 3] = rng.normal(0.0, WEIGHT_SCALE, (10, 32, 32))
    rgb_sobel = MyLinearClassifier(input_size=4096)
    rgb_sobel.W = weights.reshape(10, 4096)
    return {"RGB": rgb, "RGB+Sobel": rgb_sobel}


def evaluate(dataset, models):
    labels = np.asarray(dataset.targets)
    support = np.bincount(labels, minlength=10)
    results = {}
    for name, model in models.items():
        counts = np.zeros((10, 10), dtype=np.int64)
        probability_sums = np.zeros((10, 10))
        loss_sum = 0.0
        for start in range(0, len(dataset), 256):
            rgb = dataset.data[start:start + 256].astype(np.float32) / 255.0
            features = rgb if name == "RGB" else make_features(rgb)
            X = features.reshape(len(features), -1)
            y = labels[start:start + len(features)]
            scores = X @ model.W.T + model.b
            predictions = scores.argmax(axis=1)
            shifted_scores = scores - scores.max(axis=1, keepdims=True)
            probabilities = np.exp(shifted_scores)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            margins = np.maximum(0, scores - scores[np.arange(len(y)), y, None] + 1)
            margins[np.arange(len(y)), y] = 0
            loss_sum += float(margins.sum())
            np.add.at(counts, (y, predictions), 1)
            np.add.at(probability_sums, y, probabilities)
        correct = int(counts.diagonal().sum())
        results[name] = {
            "features": model.W.shape[1], "correct_count": correct,
            "accuracy": correct / len(dataset), "mean_hinge_loss": loss_sum / len(dataset),
            "confusion_counts": counts.tolist(), "prediction_rates": (counts / support[:, None]).tolist(),
            "mean_softmax_probabilities": (probability_sums / support[:, None]).tolist(),
        }
        print(f"{name} | 입력={model.W.shape[1]} | 정답={correct}/{len(dataset)} | "
              f"Accuracy={correct / len(dataset):.2%} | mean hinge loss={loss_sum / len(dataset):.8f}", flush=True)
    return {"seed": SEED, "trained": False, "test_size": len(dataset),
            "class_names": dataset.classes, "support": support.tolist(), "models": results}


def visualize_confusion_matrices(result):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), layout="constrained")
    fig.suptitle(f"RGB vs RGB+Sobel | random weights, seed={SEED} | no training", fontsize=16)
    labels = result["class_names"]
    for ax, (name, model) in zip(axes, result["models"].items()):
        rates = np.asarray(model["prediction_rates"])
        image = ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(10), labels, rotation=45, ha="right")
        ax.set_yticks(range(10), labels)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(f"{name} | {model['accuracy']:.2%} correct")
        for (row, column), value in np.ndenumerate(rates):
            ax.text(column, row, f"{value:.1%}", ha="center", va="center", fontsize=7,
                    color="white" if value > 0.5 else "#18344C")
    fig.colorbar(image, ax=axes, format=PercentFormatter(1), label="Prediction share", shrink=0.8)
    fig.savefig(OUTPUT_DIR / "confusion_comparison.png", dpi=140)
    plt.close(fig)


def visualize_channels(dataset):
    rgb = dataset.data[:3].astype(np.float32) / 255.0
    features = make_features(rgb)
    fig, axes = plt.subplots(3, 2, figsize=(7, 9), layout="constrained")
    fig.suptitle("RGB image and fourth channel: Sobel edge magnitude", fontsize=13)
    for index, row in enumerate(axes):
        row[0].imshow(rgb[index], interpolation="nearest")
        row[0].set_title(f"Test #{index}: {dataset.classes[dataset.targets[index]]}")
        row[1].imshow(features[index, ..., 3], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        row[1].set_title("Sobel channel")
        for ax in row:
            ax.set_axis_off()
    fig.savefig(OUTPUT_DIR / "sobel_examples.png", dpi=120)
    plt.close(fig)


def save_report(result):
    labels = result["class_names"]
    rgb, sobel = result["models"].values()
    rows = [[name, model["features"], f"{model['correct_count']:,}/{result['test_size']:,}",
             f"{model['accuracy']:.2%}", f"{model['mean_hinge_loss']:.8f}"]
            for name, model in result["models"].items()]
    sections = [
        "# RGB와 RGB+소벨 · 무작위 가중치 선형 분류 비교",
        "RGB에 경계 강도를 나타내는 소벨 채널을 하나 추가했다. 픽셀당 값은 "
        "`(R, G, B, edge)` 4개이며, 32×32 이미지 한 장을 펼치면 4,096차원이다.",
        "## 비교 조건",
        f"공식 테스트 {result['test_size']:,}장, 시드 `{SEED}`, 편향 0, 무작위 가중치의 "
        f"표준편차 `{WEIGHT_SCALE}`를 사용했다. 두 모델의 RGB 가중치는 정확히 공유하고 "
        "추가 경계 채널에 해당하는 가중치만 같은 난수 생성기에서 이어서 추출했다. "
        "과제 조건에 따라 학습 없이 `Wx+b`로 예측했다.",
        "RGB를 0~1로 변환하고 회색조 `0.299R+0.587G+0.114B`에서 소벨의 "
        "가로·세로 응답을 계산했다. `sqrt(gx²+gy²)/(4*sqrt(2))`로 경계 강도를 "
        "0~1 범위로 맞췄고, 이미지 바깥은 반사 패딩으로 처리했다.",
        "![RGB와 소벨 경계 채널](sobel_examples.png)",
        "## 결과",
        markdown_table(["입력", "특징 수", "정답 이미지 수", "정답 비율", "평균 SVM hinge loss"], rows),
        f"소벨 추가 후 정답 비율은 {100 * (sobel['accuracy'] - rgb['accuracy']):+.2f}%p 바뀌었다. "
        "이 결과는 시드 18의 무작위 가중치 한 쌍에서 얻은 관측이다. "
        "경계 채널의 가중치도 무작위이므로 경계 정보를 정답과 연결하는 규칙을 학습한 상태가 아니다. "
        "따라서 이 결과만으로 학습 후의 소벨 채널 효과를 결론 내릴 수 없다.",
        "## 10×10 혼동 행렬 비교",
        "행은 실제 클래스, 열은 예측 클래스다. 각 행은 클래스별 1,000장의 분류 비율이며 "
        "합은 100%다. 두 그림은 0~100%의 같은 파란색 눈금을 사용한다.",
        "![RGB와 RGB+소벨 혼동 행렬](confusion_comparison.png)",
    ]
    for name, model in result["models"].items():
        sections += [f"### {name} 분류 비율", markdown_table(
            ["실제 / 예측", *labels],
            [[label, *[f"{value:.2%}" for value in row]]
             for label, row in zip(labels, model["prediction_rates"])],
        )]
        prefix = "rgb" if name == "RGB" else "rgb_sobel"
        for key in ["confusion_counts", "prediction_rates", "mean_softmax_probabilities"]:
            filename = f"{prefix}_{key}.csv"
            with (OUTPUT_DIR / filename).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["true_class", *labels])
                writer.writerows([[label, *row] for label, row in zip(labels, model[key])])
    sections += [
        "## 실행",
        "저장소 루트에서 실행한다. 데이터가 필요하면 `--download`를 추가한다.",
        "```bash\npython linear_classifier_1118/CompareRGBSobel1118.py\n```",
        "[소스 코드](../CompareRGBSobel1118.py) · [원래 수치와 설정](result.json)",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="RGB와 경계 채널을 추가한 선형 분류 비교")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    dataset = CIFAR10(root=str(args.data_dir), train=False, download=args.download)
    models = make_models()
    original_weights = {name: model.W.copy() for name, model in models.items()}
    with threadpool_limits(limits=2, user_api="blas"):
        result = evaluate(dataset, models)
    for name, model in models.items():
        np.testing.assert_array_equal(model.W, original_weights[name])
        np.testing.assert_array_equal(model.b, np.zeros(10))
    OUTPUT_DIR.mkdir(exist_ok=True)
    visualize_confusion_matrices(result)
    visualize_channels(dataset)
    save_report(result)


if __name__ == "__main__":
    main()
