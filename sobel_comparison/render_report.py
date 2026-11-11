"""metrics.json에서 GitHub용 비교 그래프와 결과 표를 만든다."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
NAMES = ("RGB", "RGB+Sobel")
COLORS = ("#526884", "#127D70")
METRICS = {"accuracy": "Accuracy", "precision": "Macro Precision",
           "recall": "Macro Recall", "f1": "Macro F1"}


def performance_chart(models, config):
    fig, (test_ax, fold_ax) = plt.subplots(1, 2, figsize=(12.6, 4.6), layout="constrained")
    fig.suptitle(f"CIFAR-10 linear classifier comparison  |  L2 = {config['l2']:g}",
                 fontsize=17, fontweight="bold")
    positions = np.arange(len(METRICS))
    for index, (name, color) in enumerate(zip(NAMES, COLORS)):
        metrics = models[name]["test"]
        scores = [100 * metrics[key] for key in METRICS]
        bars = test_ax.bar(positions + (index - 0.5) * 0.35, scores, width=0.35,
                           color=color, label=name)
        test_ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        folds = models[name]["folds"]
        fold_ax.plot([f["fold"] for f in folds], [100 * f["accuracy"] for f in folds],
                     marker="o", linewidth=2, color=color, label=name)

    test_ax.set_title(f"Official test set ({config['test_size']:,} images)", loc="left")
    test_ax.set_xticks(positions, ["Accuracy", "Macro\nPrecision", "Macro\nRecall", "Macro F1"])
    test_ax.set_ylabel("Score (%)")
    test_ax.set_ylim(0, 62)
    fold_ax.set_title("5-fold validation", loc="left")
    fold_ax.set_xlabel("Fold")
    fold_ax.set_ylabel("Accuracy (%)")
    fold_ax.set_xticks(range(1, config["folds"] + 1))
    fold_ax.set_ylim(37, 51)
    for ax in (test_ax, fold_ax):
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", frameon=False)
    fig.savefig(ROOT / "performance.png", dpi=150)
    plt.close(fig)


def confusion_chart(models, labels, config):
    matrices = [np.asarray(models[name]["confusion_matrix"]) for name in NAMES]
    max_count = max(int(matrix.max()) for matrix in matrices)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.0), layout="constrained")
    fig.suptitle(f"CIFAR-10 test confusion matrices  |  L2 = {config['l2']:g}",
                 fontsize=17, fontweight="bold")
    for ax, name, matrix in zip(axes, NAMES, matrices):
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max_count)
        ax.set_title(name, loc="left", fontsize=13, fontweight="bold")
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.tick_params(length=0)
        for (row, col), count in np.ndenumerate(matrix):
            ax.text(col, row, str(count), ha="center", va="center", fontsize=8,
                    color="white" if count > max_count * 0.53 else "#173047",
                    fontweight="bold" if row == col else "normal")
    fig.colorbar(image, ax=axes, label="Images", shrink=0.76, pad=0.025)
    fig.savefig(ROOT / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
    ])


def score_row(name, rgb, sobel):
    return name, f"{rgb:.2%}", f"{sobel:.2%}", f"{100 * (sobel - rgb):+.2f}%p"


def write_readme(result):
    rgb, sobel = (result["models"][name] for name in NAMES)
    labels = result["labels"]
    config = result["config"]
    headers = ["지표", "RGB", "RGB+소벨", "차이 (소벨 − RGB)"]
    test_rows = [score_row(name, rgb["test"][key], sobel["test"][key]) for key, name in METRICS.items()]
    fold_rows = [score_row(f"Fold {a['fold']}", a["accuracy"], b["accuracy"])
                 for a, b in zip(rgb["folds"], sobel["folds"])]
    fold_rows.append(score_row("평균", rgb["cv_mean"]["accuracy"], sobel["cv_mean"]["accuracy"]))
    paired = result["paired_correctness"]
    class_rows = []
    for label, a, b in zip(labels, rgb["classes"], sobel["classes"]):
        class_rows.append([label, a["support"], *(f"{row[key]:.2%}" for row in (a, b)
                           for key in ("precision", "recall", "f1"))])
    sections = [
        "# CIFAR-10 선형 분류기 · 소벨 윤곽 채널 비교",
        f"두 모델 모두 L2 규제 강도를 `{config['l2']:g}`로 고정했습니다. "
        f"공식 테스트 {config['test_size']:,}장 기준으로 정확도가 "
        f"**{rgb['test']['accuracy']:.2%} → {sobel['test']['accuracy']:.2%}**로 바뀌었습니다.",
        "## 실험 설정",
        f"- 소프트맥스 선형 분류기, 학습률 `{config['learning_rate']:g}`, "
        f"배치 {config['batch_size']}장, {config['epochs']} epochs\n"
        f"- CIFAR-10 공식 학습 {rgb['train_size']:,}장으로 같은 계층화 5-fold 분할 "
        f"(`seed={config['seed']}`, fold당 학습 40,000장·검증 10,000장)\n"
        "- 교차 검증 후 두 모델을 각각 공식 학습 50,000장으로 다시 학습하고 "
        f"동일한 공식 테스트 {config['test_size']:,}장에 평가\n"
        f"- RGB는 {rgb['features']:,}차원, RGB+소벨은 회색조의 소벨 기울기 크기 채널을 붙인 "
        f"{sobel['features']:,}차원\n"
        "- 입력 평균은 각 fold의 학습 부분 또는 최종 학습 데이터에서 구함",
        "## 성능 그래프",
        "![공식 테스트의 네 지표와 5-fold 정확도 비교](performance.png)",
        "## 공식 테스트 성능",
        table(headers, test_rows),
        "## 5-fold 정확도",
        table(headers, fold_rows),
        "## 클래스별 성능",
        table(["클래스", "테스트 장수", "RGB Precision", "RGB Recall", "RGB F1",
               "소벨 Precision", "소벨 Recall", "소벨 F1"], class_rows),
        "## 정답과 예측의 매칭 · 혼동 행렬",
        "행은 실제 정답, 열은 모델의 예측입니다. 대각선은 올바르게 분류된 이미지 수입니다. 두 그림의 색상 눈금은 같습니다.",
        "![RGB와 RGB+소벨의 클래스별 혼동 행렬](confusion_matrices.png)",
    ]
    for name in NAMES:
        rows = [[label, *counts] for label, counts in zip(labels, result["models"][name]["confusion_matrix"])]
        sections.append(f"<details>\n<summary>{name} 혼동 행렬 표 (이미지 수)</summary>\n\n"
                        + table(["정답 / 예측", *labels], rows) + "\n\n</details>")
    sections += [
        "## 같은 이미지의 정답 여부 비교",
        table(["RGB 정답 여부", "소벨 오답", "소벨 정답"],
              [["RGB 오답", f"{paired[0][0]:,}", f"{paired[0][1]:,}"],
               ["RGB 정답", f"{paired[1][0]:,}", f"{paired[1][1]:,}"]]),
        f"소벨 채널 추가 후 기존 오답 {paired[0][1]:,}장이 정답으로 바뀌었고, "
        f"기존 정답 {paired[1][0]:,}장이 오답으로 바뀌었습니다.",
        "## 재현 방법",
        "저장소 루트에서 아래 명령을 실행합니다. `--download`는 `data/`에 CIFAR-10이 필요할 때 사용합니다.",
        "```bash\npip install -r requirements.txt\npython sobel_comparison/compare.py --download\n"
        "python sobel_comparison/render_report.py\n```",
        "[원본 측정값](metrics.json)에는 fold별 지표와 혼동 행렬의 이미지 수를 기록했습니다.",
    ]
    (ROOT / "README.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def main():
    result = json.loads((ROOT / "metrics.json").read_text(encoding="utf-8"))
    performance_chart(result["models"], result["config"])
    confusion_chart(result["models"], result["labels"], result["config"])
    write_readme(result)


if __name__ == "__main__":
    main()
