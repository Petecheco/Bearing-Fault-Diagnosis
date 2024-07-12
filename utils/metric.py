import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score


def GetOutputs(model, dataloader, device="cpu"):
    outputs = []
    y_true = []
    with torch.no_grad():
        model.to(device)
        model.eval()
        for data, labels in dataloader:
            data = data.to(device)
            outputs.extend(model(data).detach().cpu())
            y_true.extend(labels.detach().cpu())
    return outputs, y_true


def OutputToPredictions(outputs):
    probabilities = F.softmax(outputs, dim=1)
    predictions = torch.argmax(probabilities, dim=1)
    return predictions


def CalculateAccuracy(y_true, outputs):
    predictions = OutputToPredictions(outputs)
    total = len(y_true)
    correct = 0
    correct += (predictions == y_true).sum().item()
    return correct / total


def CalculatePrecision(y_true, outputs):
    predictions = OutputToPredictions(outputs)
    precision = precision_score(y_true, predictions, average='macro')
    return precision


def CalculateRecall(y_true, outputs):
    predictions = OutputToPredictions(outputs)
    recall = recall_score(y_true, predictions, average='macro')
    return recall


def CalculateF1(y_true, outputs):
    predictions = OutputToPredictions(outputs)
    score = f1_score(y_true, predictions, average="macro")
    return score


def CalculateCrossEntropy(y_true, outputs):
    return F.cross_entropy(outputs, y_true).numpy()


if __name__ == '__main__':
    outputs = torch.tensor([[0.1, 0.9],
                            [0.9, 0.1],
                            [0.1, 0.9],
                            [0.9, 0.1],
                            [0.1, 0.9],
                            [0.9, 0.1],
                            [0.1, 0.9]])
    y_true = torch.tensor((1, 1, 1, 0, 1, 0, 1))
    acc = CalculateCrossEntropy(y_true, outputs)
    print(acc)
