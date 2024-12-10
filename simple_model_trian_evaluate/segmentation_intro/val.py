import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from utils import iou_pytorch, dice_pytorch, BCE_dice 
from model import Models

def main(test_loader, model, test_dataset, loss_fn, dice_pytorch, iou_pytorch):
    """
    Evaluate the model on the test set and save results to a text file.
    
    Args:
        test_loader (DataLoader): DataLoader for the test dataset.
        model (torch.nn.Module): Trained PyTorch model.
        test_dataset (Dataset): The test dataset.
        loss_fn (function): Loss function to compute the loss.
        dice_pytorch (function): Function to compute the Dice coefficient.
        iou_pytorch (function): Function to compute the IoU metric.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Models.Unet()
    model.eval()  # Set the model to evaluation mode.
    
    running_IoU = 0.0
    running_dice = 0.0
    running_loss = 0.0
    
    with torch.no_grad():  # Disable gradient computation for evaluation.
        for i, data in enumerate(test_loader):
            img, mask = data
            img, mask = img.to(device), mask.to(device)
            
            # Get model predictions
            predictions = model(img)
            predictions = predictions.squeeze(1)  # Remove the singleton channel dimension
            
            # Compute metrics
            running_dice += dice_pytorch(predictions, mask).sum().item()
            running_IoU += iou_pytorch(predictions, mask).sum().item()
            
            # Compute loss
            loss = loss_fn(predictions, mask)
            running_loss += loss.item() * img.size(0)  # Accumulate batch loss
    
    # Compute averages
    loss = running_loss / len(test_dataset)
    dice = running_dice / len(test_dataset)
    IoU = running_IoU / len(test_dataset)
    
    # Print metrics
    result_str = f'Tests: loss: {loss:.4f} | Mean IoU: {IoU:.4f} | Dice coefficient: {dice:.4f}'
    print(result_str)
    
    # Save results to a text file
    with open("test_results.txt", "w") as file:
        file.write(result_str + "\n")


if __name__ == '__main__':
    main()
