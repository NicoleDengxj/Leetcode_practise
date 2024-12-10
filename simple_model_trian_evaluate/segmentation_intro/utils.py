import matplotlib.pyplot as plt
from PIL import Image
import torch

# from utils import iou_pytorch, dice_pytorch, BCE_dice

def iou_pytorch(predictions: torch.Tensor, labels: torch.Tensor, e: float = 1e-7):
    """Calculates Intersection over Union for a tensor of predictions"""
    predictions = torch.where(predictions > 0.5, 1, 0)
    labels = labels.byte()
    
    intersection = (predictions & labels).float().sum((1, 2))
    union = (predictions | labels).float().sum((1, 2))
    
    iou = (intersection + e) / (union + e)
    return iou

def dice_pytorch(predictions: torch.Tensor, labels: torch.Tensor, e: float = 1e-7):
    """Calculates Dice coefficient for a tensor of predictions"""
    predictions = torch.where(predictions > 0.5, 1, 0)
    labels = labels.byte()
    
    intersection = (predictions & labels).float().sum((1, 2))
    return ((2 * intersection) + e) / (predictions.float().sum((1, 2)) + labels.float().sum((1, 2)) + e)

def BCE_dice(output, target, alpha=0.01):
    bce = torch.nn.functional.binary_cross_entropy(output, target)
    soft_dice = 1 - dice_pytorch(output, target).mean()
    return bce + alpha * soft_dice



def print_train_val_loss(num_epochs,history):
    result_folder = "./result/"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)       
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), history['train_loss'], label='Training Loss', color='blue')
    plt.plot(range(1, num_epochs + 1), history['val_loss'], label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./result/train_val_loss_image.png")
    
    
def print_IOU_DICE(num_epochs, history):
    """
    Plots IoU and Dice metrics over training epochs.

    Parameters:
    - num_epochs: int, the total number of epochs
    - history: dict, contains 'val_IoU' and 'val_dice' as keys with metric values for each epoch

    Saves the plot to the './result/' directory and displays it.
    """
    # Ensure the result folder exists
    result_folder = "./result/"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    # Plot IoU and Dice metrics
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), history['val_IoU'], label='Validation Mean Jaccard Index (IoU)', color='blue')
    plt.plot(range(1, num_epochs + 1), history['val_dice'], label='Validation Dice Coefficient', color='red')
    
    # Add titles and labels
    plt.title("IoU and Dice Coefficient Over Epochs", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plot_path = os.path.join(result_folder, "iou_dice_plot.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to '{plot_path}'.")



def training_loop(epochs, model, train_loader, valid_loader, optimizer, loss_fn, lr_scheduler,save_path="best_model.pth"):
    history = {'train_loss': [], 'val_loss': [], 'val_IoU': [], 'val_dice': []}
    early_stopping = EarlyStopping(patience=7)
    best_val_loss = float('inf') 
    
    
    
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        running_loss = 0
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            img, mask = data
            img, mask = img.to(device), mask.to(device)
            predictions = model(img)
            predictions = predictions.squeeze(1)
            loss = loss_fn(predictions, mask)
            running_loss += loss.item() * img.size(0)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        model.eval()
        with torch.no_grad():
            running_IoU = 0
            running_dice = 0
            running_valid_loss = 0
            for i, data in enumerate(valid_loader):
                img, mask = data
                img, mask = img.to(device), mask.to(device)
                predictions = model(img)
                predictions = predictions.squeeze(1)
                running_dice += dice_pytorch(predictions, mask).sum().item()
                running_IoU += iou_pytorch(predictions, mask).sum().item()
                loss = loss_fn(predictions, mask)
                running_valid_loss += loss.item() * img.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        val_loss = running_valid_loss / len(valid_loader.dataset)
        val_dice = running_dice / len(valid_loader.dataset)
        val_IoU = running_IoU / len(valid_loader.dataset)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_IoU'].append(val_IoU)
        history['val_dice'].append(val_dice)
        print(f'Epoch: {epoch}/{epochs} | Training loss: {train_loss} | Validation loss: {val_loss} | Validation Mean IoU: {val_IoU} '
         f'| Validation Dice coefficient: {val_dice}')
        
        lr_scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at epoch {epoch} with validation loss: {val_loss:.4f}")
        
        
        
        
        if early_stopping(val_loss, model):
            early_stopping.load_weights(model)
            break
    model.eval()
    return history















def load_and_display_image(filepath, title="Image"):        
    try:
        # Load the image
        img = Image.open(filepath).convert("RGB")
        
        # Convert to a PyTorch tensor
        transform = transforms.ToTensor()
        image_tensor = transform(img)  # Shape: (C, H, W)
        
        # Get resolution
        resolution = f"{img.width}x{img.height}"  # Width x Height

        # Print file path and resolution
        print(f"Filepath: {filepath}")
        print(f"Resolution: {resolution}")
        
        # Display the image
        plt.figure(figsize=(6, 6))
        plt.imshow(image_tensor.permute(1, 2, 0))  # Convert (C, H, W) to (H, W, C)
        plt.title(f"{title}\nResolution: {resolution}")
        plt.axis("off")
        plt.show()
    except Exception as e:
        print(f"Error loading image: {e}")
#load_and_display_image(filename, title="Sample RGB Image")
