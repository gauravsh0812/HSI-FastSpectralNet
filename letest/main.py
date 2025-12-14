import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from data_loader import load_pavia_university, preprocess_data, PaviaUniversityDataset
from model import newFastViT
from train import setup_trainer
from utils import *

if __name__ == "__main__":
    image_file = "/content/PaviaU.mat"
    gt_file = "/content/PaviaU_gt.mat"

    image_data, ground_truth = load_pavia_university(image_file, gt_file)
    spatial_spectral_data, y, label_encoder = preprocess_data(image_data, ground_truth)

    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
    train_dataset = PaviaUniversityDataset(spatial_spectral_data[train_idx], y[train_idx])
    test_dataset = PaviaUniversityDataset(spatial_spectral_data[test_idx], y[test_idx])

    model = newFastViT(num_channels=103, num_classes=len(np.unique(y)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    trainer = setup_trainer(model, train_dataset, test_dataset)
    trainer.train()
    eval_results = trainer.evaluate()
    print("Evaluation Results:", eval_results)

    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)

    print(f"OA: {overall_accuracy(y[test_idx], y_pred):.4f}")
    print(f"AA: {average_accuracy(y[test_idx], y_pred):.4f}")
    print(f"Kappa: {kappa_coefficient(y[test_idx], y_pred):.4f}")
    f1, precision, recall = calculate_f1_precision_recall(y[test_idx], y_pred)
    print(f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"Latency per image: {calculate_latency_per_image(model, test_loader, device):.4f} ms")
    print(f"Throughput: {calculate_throughput(model, test_loader, device):.2f} samples/sec")
    print(f"Parameters: {count_model_parameters(model):.2f} M")
    print(f"GFLOPs: {calculate_gflops(model, train_dataset, device):.2f}")

    cm = confusion_matrix(y[test_idx], y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
