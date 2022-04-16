import json
import pandas as pd
import matplotlib.pyplot as plt


def parse_results_dict(results_dict):
    epochs = None
    temp_dict = {}
    for key, values in results_dict.items():
        if "epochs" not in temp_dict:
            temp_dict["epochs"] = values["steps"]
            temp_dict["timestamps"] = values["timestamps"]
        temp_dict[key] = values["values"]
    return pd.DataFrame(temp_dict)


def get_results(path_list):
    results_frames = []
    for path in path_list:
        with open(path, "r") as f:
            results_dict = json.load(f)
        results_frames.append(parse_results_dict(results_dict))
    return pd.concat(results_frames)


def visualize_loss(results):
    metric_pairs = [('meters/loss_D/train', 'meters/loss_D/test'),
                    ('meters/loss_G/train', 'meters/loss_G/test'),
                    ('meters/loss_MSE/train', 'meters/loss_MSE/test')]

    fig, axes = plt.subplots(3, 1, figsize=(10, 5))

    for i, (train, test) in enumerate(metric_pairs):
        axes[i].plot(results["epochs"], results[train], label="train")
        axes[i].plot(results["epochs"], results[test], label="test")
        axes[i].set_title(train.split("/")[1])
        axes[i].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    gdrive = "/content/drive/MyDrive/CS7643-GroupProject/teammates/Kasey/celebA_removePixel_p95"

    path_list = []
    for run_id in [2, 5, 6]:
        path_list.append(f"{gdrive}/{run_id}/metrics.json")

    results = get_results(path_list)
    visualize_loss(results)
     