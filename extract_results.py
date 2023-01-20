import os
import pandas as pd
import numpy as np
import json

if __name__ == '__main__':
    # type = "noise_2"
    # type = "noise_3"
    type = "new_noise_fix_batch_16_e-6"
    base_path = "/home/is/gabriel-he/pycharm-upload/SimpleNER/results/" + type
    folders = os.listdir(base_path)
    output = {
        "noise": [],
        "train_loss": [],
        "train_loss_std": [],
        "val_loss": [],
        "val_loss_std": [],
        "val_f1": [],
        "val_f1_std": [],
        "val_strawberry": [],
        "val_strawberry_std": [],
        "test_precision": [],
        "test_precision_std": [],
        "test_recall": [],
        "test_recall_std": [],
        "test_f1": [],
        "test_f1_std": [],
        "test_strawberry": [],
        "test_strawberry_std": [],
        "test_precision_relaxed": [],
        "test_precision_relaxed_std": [],
        "test_recall_relaxed": [],
        "test_recall_relaxed_std": [],
        "test_f1_relaxed": [],
        "test_f1_relaxed_std": [],
        "test_specificity": [],
        'accuracy': [],
    }
    for folder in sorted(folders):
        path = os.path.join(base_path, folder)
        noise = folder.split("_")[1]
        models = os.listdir(path)

        noise_data = {
            "noise": noise,
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
            "val_strawberry": [],
            "test_precision": [],
            "test_recall": [],
            "test_f1": [],
            "test_strawberry": [],
            "test_precision_relaxed": [],
            "test_recall_relaxed": [],
            "test_f1_relaxed": [],
            "test_specificity": [],
            'accuracy': [],
        }

        for model in models:
            try:
                model_path = os.path.join(path, model)

                # Get training metrics
                data = json.loads(open(os.path.join(model_path, "training_metrics.txt")).read())
                noise_data["train_loss"].append(data["loss"])
                noise_data["val_loss"].append(data["val_loss"])

                # Get validation metrics
                data = json.loads(open(os.path.join(model_path, "test_metrics.txt")).read())
                noise_data["val_f1"].append(data["f1"])
                noise_data["val_strawberry"].append(data["strawberry"])

                # Get test metrics
                data = json.loads(open(os.path.join(model_path, "output_last/eval_results.txt")).read())
                noise_data["test_precision"].append(data["precision"])
                noise_data["test_recall"].append(data["recall"])
                noise_data["test_f1"].append(data["f1"])
                noise_data["test_strawberry"].append(data["strawberry"])
                noise_data["test_precision_relaxed"].append(data["overall_precision_relaxed"])
                noise_data["test_recall_relaxed"].append(data["overall_recall_relaxed"])
                noise_data["test_f1_relaxed"].append(data["overall_f1_relaxed"])
                noise_data["test_specificity"].append(data["specificity"])
                noise_data["accuracy"].append(data["accuracy"])
            except:
                pass

        if not noise_data["test_f1"]:
            continue

        # Calculate average and standard deviation
        output["noise"].append(noise_data["noise"])
        output["train_loss"].append(np.mean(noise_data["train_loss"]))
        output["train_loss_std"].append(np.std(noise_data["train_loss"]))
        output["val_loss"].append(np.mean(noise_data["val_loss"]))
        output["val_loss_std"].append(np.std(noise_data["val_loss"]))
        output["val_f1"].append(np.mean(noise_data["val_f1"]))
        output["val_f1_std"].append(np.std(noise_data["val_f1"]))
        output["val_strawberry"].append(np.mean(noise_data["val_strawberry"]))
        output["val_strawberry_std"].append(np.std(noise_data["val_strawberry"]))
        output["test_precision"].append(np.mean(noise_data["test_precision"]))
        output["test_precision_std"].append(np.std(noise_data["test_precision"]))
        output["test_recall"].append(np.mean(noise_data["test_recall"]))
        output["test_recall_std"].append(np.std(noise_data["test_recall"]))
        output["test_f1"].append(np.mean(noise_data["test_f1"]))
        output["test_f1_std"].append(np.std(noise_data["test_f1"]))
        output["test_strawberry"].append(np.mean(noise_data["test_strawberry"]))
        output["test_strawberry_std"].append(np.std(noise_data["test_strawberry"]))
        output["test_precision_relaxed"].append(np.mean(noise_data["test_precision_relaxed"]))
        output["test_precision_relaxed_std"].append(np.std(noise_data["test_precision_relaxed"]))
        output["test_recall_relaxed"].append(np.mean(noise_data["test_recall_relaxed"]))
        output["test_recall_relaxed_std"].append(np.std(noise_data["test_recall_relaxed"]))
        output["test_f1_relaxed"].append(np.mean(noise_data["test_f1_relaxed"]))
        output["test_f1_relaxed_std"].append(np.std(noise_data["test_f1_relaxed"]))
        output["test_specificity"].append(np.mean(noise_data["test_specificity"]))
        output['accuracy'].append(np.mean(noise_data['accuracy']))
    df = pd.DataFrame(output)
    df.to_excel(type + "2.xlsx", index=False)
