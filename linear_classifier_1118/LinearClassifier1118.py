from pathlib import Path
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CIFAR10


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent
SEED = 18
WEIGHT_SCALE = 0.01
DELTA = 1.0


class MyLinearClassifier:
    def __init__(self, input_size=3072, num_classes=10):
        rng = np.random.default_rng(SEED)
        self.W = rng.normal(0.0, WEIGHT_SCALE, (num_classes, input_size))
        self.b = np.zeros(num_classes)

    def score(self, x):
        return self.W @ x + self.b

    def predict(self, x):
        return int(np.argmax(self.score(x)))


def svm_hinge_loss(scores, correct_label):
    margins = np.maximum(0.0, scores - scores[correct_label] + DELTA)
    margins[correct_label] = 0.0
    return float(np.sum(margins)), margins


def softmax(scores):
    shifted_scores = scores - np.max(scores)
    exp_scores = np.exp(shifted_scores)
    return exp_scores / np.sum(exp_scores)


def class_probability_tables(dataset, model):
    num_classes = len(dataset.classes)
    counts = np.zeros((num_classes, num_classes), dtype=np.int64)
    probability_sums = np.zeros((num_classes, num_classes))
    labels = np.asarray(dataset.targets)
    support = np.bincount(labels, minlength=num_classes)

    for start in range(0, len(dataset), 256):
        images = dataset.data[start:start + 256]
        X = images.reshape(len(images), -1).astype(np.float32) / 255.0
        scores = X @ model.W.T + model.b
        predictions = np.argmax(scores, axis=1)
        shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
        probabilities = np.exp(shifted_scores)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        true_labels = labels[start:start + len(images)]
        np.add.at(counts, (true_labels, predictions), 1)
        np.add.at(probability_sums, true_labels, probabilities)

    prediction_rates = counts / support[:, None]
    mean_probabilities = probability_sums / support[:, None]
    assert np.allclose(prediction_rates.sum(axis=1), 1.0)
    assert np.allclose(mean_probabilities.sum(axis=1), 1.0)
    return {
        "sample_count": len(dataset), "support": support.tolist(),
        "confusion_counts": counts.tolist(),
        "prediction_rates": prediction_rates.tolist(),
        "mean_softmax_probabilities": mean_probabilities.tolist(),
    }


def select_examples(dataset, model):
    selected = [0, 1, 2, 3]
    first_correct = None
    first_wrong = None

    for index, image in enumerate(dataset.data):
        x = image.reshape(-1).astype(np.float32) / 255.0
        prediction = model.predict(x)
        if prediction == dataset.targets[index] and first_correct is None:
            first_correct = index
        if prediction != dataset.targets[index] and first_wrong is None:
            first_wrong = index
        if first_correct is not None and first_wrong is not None:
            break

    for index in (first_correct, first_wrong):
        if index is not None and index not in selected:
            selected.append(index)
    return selected, first_correct, first_wrong


def visualize_examples(dataset, examples):
    columns = 3
    rows = (len(examples) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(10, 3.5 * rows), squeeze=False,
                             layout="constrained")
    fig.suptitle(f"CIFAR-10 examples | random weights, seed={SEED}", fontsize=15)
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, example in zip(axes.flat, examples):
        index = example["index"]
        ax.imshow(dataset.data[index], interpolation="nearest")
        state = "Correct" if example["correct"] else "Incorrect"
        ax.set_title(f"Test #{index} | {state}\nTrue: {example['true_class']}\n"
                     f"Predicted: {example['predicted_class']}",
                     fontsize=11, color="#177245" if example["correct"] else "#AD343E")
    fig.savefig(OUTPUT_DIR / "examples.png", dpi=130)
    plt.close(fig)


def visualize_weights(model, class_names):
    weights = model.W.reshape(10, 32, 32, 3)
    max_abs = np.max(np.abs(weights))
    displayed = 0.5 + weights / (2 * max_abs)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.4), layout="constrained")
    fig.suptitle("Random class weights | 32 x 32 x 3 | shared display scale", fontsize=15)
    for ax, image, class_name in zip(axes.flat, displayed, class_names):
        ax.imshow(image, interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(class_name)
        ax.set_axis_off()
    fig.savefig(OUTPUT_DIR / "weights.png", dpi=130)
    plt.close(fig)


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    return "\n".join(lines)


def write_report(result):
    examples = result["examples"]
    class_tables = result["class_tables"]
    class_names = result["class_names"]
    overview = [[row["index"], row["true_class"], row["predicted_class"],
                 "정답" if row["correct"] else "오답", f"{row['hinge_loss']:.8f}",
                 f"{row['probability_sum']:.12f}"] for row in examples]
    sections = [
        "# 선형 분류 과제 · 11월 18일",
        "CIFAR-10 이미지에서 무작위로 초기화한 선형 분류기의 클래스 점수, "
        "다중 클래스 SVM hinge loss, Softmax 확률을 확인한다.",
        "## 입력과 가중치 초기화",
        "공식 테스트 세트의 RGB 이미지를 사용했다. 이미지 한 장의 크기는 32×32×3이다. "
        "`torchvision`이 반환한 `(높이, 너비, RGB)` 순서의 배열을 길이 3,072로 펼치고 "
        "픽셀값을 255로 나누어 0~1 범위로 바꿨다.",
        f"`np.random.default_rng({SEED})`로 난수를 고정했다. "
        f"가중치 `W`는 평균 0, 표준편차 {WEIGHT_SCALE}의 정규분포에서 뽑았고, "
        "편향 `b`는 0으로 설정했다. `W.shape=(10, 3072)`, `b.shape=(10,)`이며 "
        "행 번호가 클래스 번호에 해당한다. 초기화한 가중치와 편향을 사용해 계산했다.",
        "선형 분류기는 `f(x)=Wx+b`로 각 클래스의 점수를 계산하고 `argmax`로 예측한다. "
        "이 구현은 무작위 초기 가중치를 사용하는 선형 분류기의 순전파 과정이다.",
        "## 실제 클래스 × 예측 클래스 · 10×10 혼동 행렬",
        f"공식 테스트 {class_tables['sample_count']:,}장을 같은 시드 {SEED}의 가중치로 예측했다. "
        "각 실제 클래스에는 1,000장이 있다. 행은 실제 클래스, 열은 `argmax`로 선택한 예측 클래스다. "
        "각 칸은 해당 실제 클래스의 이미지 중 그 열로 분류된 비율이며 행의 합은 100%다. "
        "`automobile`은 자동차(CAR), `airplane`은 비행기, `ship`은 배에 해당한다.",
        markdown_table(
            ["실제 / 예측", *class_names],
            [[name, *[f"{value:.2%}" for value in row]]
             for name, row in zip(class_names, class_tables["prediction_rates"])],
        ),
        "예를 들어 `automobile` 행의 `airplane` 열은 실제 자동차 1,000장 중 비행기로 "
        "분류된 비율이다. 무작위 가중치에서는 몇몇 클래스에 예측이 집중될 수 있다.",
        "[분류 비율 CSV](prediction_rates.csv) · [원래 이미지 수 CSV](confusion_counts.csv)",
        "## 실제 클래스 × 출력 클래스 · 평균 Softmax 확률",
        "각 실제 클래스의 이미지 1,000장에 대해 Softmax 확률을 구한 뒤 클래스별로 평균했다. "
        "행의 10개 값을 합하면 100%다. 위 혼동 행렬은 최고 점수 클래스 하나에 집계하고, "
        "이 표는 이미지마다 출력된 10개 확률을 모두 반영한다. 아래 수치는 반올림해 표시했다.",
        markdown_table(
            ["실제 / 출력", *class_names],
            [[name, *[f"{value:.2%}" for value in row]]
             for name, row in zip(class_names, class_tables["mean_softmax_probabilities"])],
        ),
        "[평균 Softmax 확률 CSV](mean_softmax_probabilities.csv)",
        "## 클래스 점수와 예측",
        "```python\nscores = W @ x + b\nprediction = np.argmax(scores)\n```",
        "먼저 테스트 인덱스 0~3을 선택했다. 같은 가중치로 테스트 이미지를 앞에서부터 "
        "확인해 처음 만난 정답 사례와 오답 사례가 목록에 없으면 추가했다. "
        "아래 평균 손실은 이 선택 목록의 이미지들을 각각 한 번씩 사용해 계산했다.",
        "![선택한 이미지의 실제 클래스와 예측](examples.png)",
        markdown_table(["테스트 인덱스", "실제 클래스", "최고 점수 클래스", "분류 결과",
                        "SVM hinge loss", "Softmax 확률 합"], overview),
        "## 클래스별 점수 비교",
        "행은 클래스, 열은 선택한 테스트 이미지다. 각 열에서 가장 높은 점수의 클래스가 예측 결과다.",
        markdown_table(
            ["클래스 번호", "클래스", *[f"테스트 #{row['index']}" for row in examples]],
            [[index, name, *[f"{row['scores'][index]:.8f}" for row in examples]]
             for index, name in enumerate(result["class_names"])],
        ),
        "## 클래스별 Softmax 확률 비교",
        "각 열의 10개 확률을 더하면 1이다. 아래 수치는 반올림해 표시했다.",
        markdown_table(
            ["클래스 번호", "클래스", *[f"테스트 #{row['index']}" for row in examples]],
            [[index, name, *[f"{row['probabilities'][index]:.8f}" for row in examples]]
             for index, name in enumerate(result["class_names"])],
        ),
        "## SVM hinge loss와 Softmax 계산",
        "정답 클래스가 `y`, 클래스 점수가 `s`, 마진이 `delta=1`일 때 이미지별 손실은 다음과 같다.",
        r"$$L_i=\sum_{j\ne y_i}\max(0,\ s_j-s_{y_i}+1)$$",
        "정답 클래스 자신은 합에서 제외했다. 이 과제에서는 클래스별 마진의 합을 "
        "사용했다. 모든 이미지의 평균은 이미지별 손실 합을 선택 이미지 수로 나눠 구했다.",
        f"**선택한 {len(examples)}개 이미지의 평균 SVM hinge loss = {result['mean_hinge_loss']:.8f}**",
        "Softmax는 각 클래스 점수의 지수값을 전체 클래스 지수값의 합으로 나눈다. "
        "수치 안정성을 위해 모든 점수에서 최고 점수를 뺀 뒤 계산했다.",
        r"$$p_j=\frac{\exp(s_j-\max_k s_k)}{\sum_k\exp(s_k-\max_l s_l)}$$",
        "각 이미지의 확률 합을 `np.isclose(sum, 1.0)`으로 확인했다. "
        "Softmax는 점수의 순서를 유지하므로 최고 점수 클래스와 최고 확률 클래스가 같다. "
        "확률 합이 1인 것은 정규화 결과이며, 예측이 정답인지와는 별개다.",
        "## 예시별 10개 클래스의 점수와 확률",
    ]
    for example in examples:
        rows = [[name, f"{s:.8f}", f"{p:.8f}", f"{m:.8f}"]
                for name, s, p, m in zip(result["class_names"], example["scores"],
                                         example["probabilities"], example["margins"])]
        sections.extend([
            f"### 테스트 #{example['index']} · 실제 {example['true_class']} → 예측 {example['predicted_class']}",
            markdown_table(["클래스", "원래 점수", "Softmax 확률", "hinge 마진 기여"], rows),
            f"손실 `{example['hinge_loss']:.8f}`, 확률 합 `{example['probability_sum']:.12f}`, "
            f"합이 1인지 확인 `{example['sum_is_one']}`.",
        ])
    sections.append("## 올바른 예측과 잘못된 예측")
    for index_key, description in [("first_correct", "처음 찾은 정답 사례"), ("first_wrong", "처음 찾은 오답 사례")]:
        index = result[index_key]
        if index is None:
            sections.append(f"{description}는 이 초기 가중치로 검색한 범위에서 발견되지 않았다.")
        else:
            example = next(row for row in examples if row["index"] == index)
            sections.append(f"{description}는 테스트 #{index}이다. 실제 클래스는 "
                            f"`{example['true_class']}`, 최고 점수 클래스는 `{example['predicted_class']}`이다.")
    sections.extend([
        "정답 사례는 무작위 가중치가 우연히 실제 클래스에 최고 점수를 준 경우다. "
        "오답 사례는 다른 클래스의 무작위 가중치와 입력의 내적이 더 크게 나온 경우다. "
        "점수와 손실을 계산하는 것만으로 가중치가 수정되지는 않는다. "
        "정답 사례가 있어도 그 클래스의 형태를 학습했다고 해석할 수 없다.",
        "정답으로 분류했더라도 다른 클래스보다 점수가 1 이상 높지 않으면 hinge loss가 "
        "양수일 수 있다. argmax로 고르는 예측과 마진까지 요구하는 손실의 조건은 다르다.",
        "## 클래스별 가중치 시각화",
        "![클래스별 무작위 가중치를 RGB 이미지로 시각화](weights.png)",
        "각 클래스의 가중치 행을 `(32, 32, 3)`으로 복원했다. 시각화할 때는 전체 가중치의 "
        "최대 절댓값 `a`를 사용해 `0.5 + W / (2*a)`로 0~1 범위에 표시했다. "
        "모든 클래스에 같은 범위를 적용하며, 이 변환은 그림을 표시할 때만 사용한다.",
        "가중치는 이미지와 정답의 관계를 이용해 갱신한 값이 아니라 정규분포에서 추출한 값이다. "
        "따라서 비행기의 날개나 자동차의 윤곽처럼 클래스와 일관되게 연결되는 학습된 특징이 "
        "나타나지 않고 무작위 색상 패턴이 보인다. 클래스 이름은 가중치 행에 붙인 라벨이다.",
        "## 실행 방법과 결과 파일",
        "저장소 루트에서 실행한다. 기존 `data/`의 CIFAR-10을 사용하며, 필요한 경우 "
        "`--download` 옵션으로 공식 데이터를 내려받을 수 있다.",
        "```bash\npip install -r requirements.txt\npython linear_classifier_1118/LinearClassifier1118.py\n"
        "python linear_classifier_1118/LinearClassifier1118.py --download\n```",
        "[소스 코드](LinearClassifier1118.py) · [실행 결과](result.txt) · [원래 수치](result.json)",
        "## 데이터 출처",
        "Alex Krizhevsky, Learning Multiple Layers of Features from Tiny Images, 2009. "
        "[CIFAR-10 공식 페이지](https://cave.cs.toronto.edu/kriz/cifar.html)",
    ])
    (OUTPUT_DIR / "README.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="무작위 가중치 선형 분류기")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    dataset = CIFAR10(root=str(args.data_dir), train=False, download=args.download)
    model = MyLinearClassifier()
    initial_W = model.W.copy()
    initial_b = model.b.copy()
    indices, first_correct, first_wrong = select_examples(dataset, model)

    examples = []
    lines = [f"seed={SEED}, W standard deviation={WEIGHT_SCALE}, delta={DELTA}",
             f"W.shape={model.W.shape}, b.shape={model.b.shape}", f"test indices={indices}"]
    for index in indices:
        x = dataset.data[index].reshape(-1).astype(np.float32) / 255.0
        scores = model.score(x)
        prediction = int(np.argmax(scores))
        correct_label = dataset.targets[index]
        loss, margins = svm_hinge_loss(scores, correct_label)
        probabilities = softmax(scores)
        probability_sum = float(np.sum(probabilities))
        assert np.isclose(probability_sum, 1.0)
        assert int(np.argmax(probabilities)) == prediction
        example = {
            "index": index, "true_label": correct_label, "predicted_label": prediction,
            "true_class": dataset.classes[correct_label], "predicted_class": dataset.classes[prediction],
            "correct": prediction == correct_label, "scores": scores.tolist(),
            "probabilities": probabilities.tolist(), "margins": margins.tolist(),
            "hinge_loss": loss, "probability_sum": probability_sum,
            "sum_is_one": bool(np.isclose(probability_sum, 1.0)),
        }
        examples.append(example)
        lines += [f"\nTest #{index}: true={example['true_class']}, predicted={example['predicted_class']}, "
                  f"correct={example['correct']}", "class          score         probability    hinge margin"]
        for name, s, p, m in zip(dataset.classes, scores, probabilities, margins):
            lines.append(f"{name:12s} {s: .8f}    {p:.8f}     {m:.8f}")
        lines += [f"SVM hinge loss={loss:.8f}",
                  f"Probability sum={probability_sum:.12f}, isclose(1)={example['sum_is_one']}"]

    class_tables = class_probability_tables(dataset, model)
    assert np.array_equal(model.W, initial_W)
    assert np.array_equal(model.b, initial_b)
    result = {"seed": SEED, "weight_scale": WEIGHT_SCALE, "delta": DELTA,
              "W_shape": list(model.W.shape), "b_shape": list(model.b.shape),
              "numpy_version": np.__version__, "class_names": dataset.classes,
              "first_correct": first_correct, "first_wrong": first_wrong,
              "examples": examples, "class_tables": class_tables,
              "mean_hinge_loss": float(np.mean([row["hinge_loss"] for row in examples]))}
    lines += [f"\nMean SVM hinge loss ({len(examples)} examples)={result['mean_hinge_loss']:.8f}",
              "W and b are identical to their initial values."]
    for title, key in [("10 x 10 prediction rates", "prediction_rates"),
                       ("10 x 10 mean Softmax probabilities", "mean_softmax_probabilities")]:
        lines += ["\n" + title, markdown_table(
            ["True / Output", *dataset.classes],
            [[name, *[f"{value:.2%}" for value in row]]
             for name, row in zip(dataset.classes, class_tables[key])],
        )]
    output = "\n".join(lines) + "\n"
    print(output)
    (OUTPUT_DIR / "result.txt").write_text(output, encoding="utf-8")
    (OUTPUT_DIR / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for filename, key in [("confusion_counts.csv", "confusion_counts"),
                          ("prediction_rates.csv", "prediction_rates"),
                          ("mean_softmax_probabilities.csv", "mean_softmax_probabilities")]:
        with (OUTPUT_DIR / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["true_class", *dataset.classes])
            for name, row in zip(dataset.classes, class_tables[key]):
                writer.writerow([name, *row])
    visualize_examples(dataset, examples)
    visualize_weights(model, dataset.classes)
    write_report(result)


if __name__ == "__main__":
    main()
