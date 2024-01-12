import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm
import numpy as np
class Validation:
    def __init__(self, opt):
        self.device = opt['device']
        self.num_classes = opt['net']['num_classes']
    def validate(self, model, dataloader, criterion):
        model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0.0

        # Use tqdm for the progress bar
        with torch.no_grad():
            with tqdm(total=len(dataloader), desc="Validation") as pbar:
            # Iterate through the DataLoader
                for inputs, targets in dataloader:
                # Move data to the device
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                    outputs = model(inputs)

                # Assuming softmax activation for multi-class classification
                    predictions = F.softmax(outputs, dim=1)

                # Append predictions and targets to lists
                    all_preds.extend(predictions.cpu().detach().numpy())
                    all_targets.extend(targets.cpu().detach().numpy())

                # Calculate loss
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                # Update tqdm
                    pbar.update(1)
                    del inputs
                    del targets
   
        # Convert predictions to class indices
        predicted_classes = [np.argmax(pred) for pred in all_preds]
        all_targets = [np.argmax(pred) for pred in all_targets]

        # Calculate precision, recall, and F1 score
        precision = precision_score(all_targets, predicted_classes, average=None, labels=range(self.num_classes), zero_division=1)
        recall = recall_score(all_targets, predicted_classes, average=None, labels=range(self.num_classes), zero_division=1)
        f1 = f1_score(all_targets, predicted_classes, average=None, labels=range(self.num_classes), zero_division=1)
        accuracy = accuracy_score(all_targets, predicted_classes)

        # Calculate average loss
        average_loss = total_loss / len(dataloader)

        model.train()
        return precision, recall, f1, average_loss,accuracy
