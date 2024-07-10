import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score


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
