import numpy as np
import time
import torch
from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score, precision_score, recall_score
from thop import profile

def calculate_latency_per_image(model, data_loader, device):
    model.eval()
    total_time, total_images = 0, 0
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['x'].to(device)
            batch_size = inputs.shape[0]
            total_images += batch_size
            start_time = time.time()
            _ = model(inputs)
            total_time += (time.time() - start_time)
    return (total_time / total_images) * 1000

def calculate_throughput(model, data_loader, device):
    model.eval()
    total_samples, total_time = 0, 0
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['x'].to(device)
            batch_size = inputs.size(0)
            start_time = time.time()
            _ = model(inputs)
            total_time += time.time() - start_time
            total_samples += batch_size
    return total_samples / total_time

def overall_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def average_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    return np.nanmean(class_accuracies)

def kappa_coefficient(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred)

def calculate_f1_precision_recall(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    return f1, precision, recall

def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000

def calculate_gflops(model, dataset, device):
    sample = dataset[0]['x'].unsqueeze(0).to(device)
    flops, _ = profile(model, inputs=(sample,), verbose=False)
    return flops / 1e9
