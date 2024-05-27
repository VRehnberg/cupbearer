import json
from glob import glob

import pandas as pd

ds = []
# e.g. PreActResNet/GTSRB/Adversarial/2349078/detector/QuantumEntropyDetector/2351607/
for eval_path in glob("*/*/*/*/detector/*/*/eval.json"):
    model, dataset, backdoor, _, _, detector, _, _ = eval_path.split("/")

    if dataset not in ["MNIST", "GTSRB", "CIFAR10"]:
        print("Skipping", eval_path)
        continue
    if detector == "AbstractionDetector":
        print("Skipping", eval_path)
        continue

    if detector.endswith("Detector"):
        detector = detector[: -len("Detector")]
    if detector == "LocallyConsistentAbstraction":
        detector = "LCA"
    if detector == "AutoencoderAbstraction":
        detector = "Autoencoder"
    if detector == "FinetuningAnomaly":
        detector = "Finetuning"

    if backdoor.endswith("Backdoor"):
        backdoor = backdoor[: -len("Backdoor")]
    if backdoor.endswith("Pixel"):
        backdoor = backdoor[: -len("Pixel")]

    data = {
        "Model": model,
        "Dataset": dataset,
        "Backdoor": backdoor,
        "Detector": detector,
    }
    with open(eval_path) as f:
        data.update(json.load(f))
    ds.append(data)

df = pd.DataFrame(ds)
print(df.groupby(["Model", "Dataset", "Backdoor", "Detector"]).mean())
print(df["Model"].unique())
print(df["Dataset"].unique())
print(df["Backdoor"].unique())
print(df["Detector"].unique())


import matplotlib.pyplot as plt  # noqa
import seaborn as sns  # noqa

sns.color_palette("colorblind")
g = sns.FacetGrid(
    data=df,
    row="Model",
    col="Dataset",
    ylim=[0, 1],
    legend_out=True,
    col_order=["MNIST", "GTSRB", "CIFAR10"],
    row_order=["PreActResNet", "CNN", "MLP"],
)


def add_line(**kwargs):
    plt.axhline(0.5, color="grey", linestyle="dashed")


g.map(add_line)
g.map(
    sns.barplot,
    "Backdoor",
    "AUC_ROC",
    "Detector",
    order=["Adversarial", "Corner", "Noise", "Wanet"],
    hue_order=[
        "Mahalanobis",
        "SpectralSignature",
        "QuantumEntropy",
        "LCA",
        "Autoencoder",
        "Finetuning",
    ],
    palette="colorblind",
)
g.add_legend()
g.savefig("results.png", dpi=300, pad_inches=0.2)

g = sns.FacetGrid(
    data=df,
    col="Model",
    row="Dataset",
    ylim=[0, 1],
    legend_out=True,
    row_order=["CIFAR10"],
    col_order=["PreActResNet", "CNN"],
)


def add_line(**kwargs):
    plt.axhline(0.5, color="grey", linestyle="dashed")


g.map(add_line)
g.map(
    sns.barplot,
    "Backdoor",
    "AUC_ROC",
    "Detector",
    order=["Adversarial", "Corner", "Noise", "Wanet"],
    hue_order=[
        "Mahalanobis",
        "SpectralSignature",
        "QuantumEntropy",
        "LCA",
        "Autoencoder",
    ],
    palette="colorblind",
)
g.add_legend()
g.savefig("results_cleaned.png", dpi=900, pad_inches=0.2)
plt.show()
