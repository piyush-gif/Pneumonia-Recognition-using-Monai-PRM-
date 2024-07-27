import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from model import CNNModel
from early_stopping import EarlyStopping
from utils import train_loader, val_loader, test_loader  
import seaborn as sns

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)

# Initialize early stopping
early_stopping = EarlyStopping(patience=3, verbose=True)

criteria = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize the StepLR scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Number of epochs
num_epochs = 10
total_batches = len(train_loader)


# Define the validate_model function before the training loop
def validate_model(model, val_loader, criteria, device):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criteria(outputs.squeeze(), labels.float())
            val_loss += loss.item()
            predictions = outputs.squeeze().round()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    avg_val_loss = val_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {accuracy:.4f}')
    return avg_val_loss, accuracy  # Return the average validation loss and accuracy


def train_model(model, train_loader, val_loader, criteria, optimizer, scheduler, device, num_epochs=10):
    # Step 1: Clear the training_results.txt file at the beginning of the training
    with open('training_results.txt', 'w') as f:
        f.write("")  # This effectively clears the file

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        total_batches = len(train_loader)
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criteria(outputs.squeeze(), labels.float())
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Update the learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Step 2: Append the epoch results to the training_results.txt file
        epoch_result = f'Epoch [{epoch+1}/{num_epochs}] completed. Loss: {running_loss/total_batches:.4f}. LR: {current_lr:.4f}\n'
        with open('training_results.txt', 'a') as f:
            f.write(epoch_result)
        
        avg_val_loss = validate_model(model, val_loader, criteria, device)
        print(epoch_result + f'Validation Loss: {avg_val_loss:.4f}')
        
        # Check if the current model is the best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved new best model")
            
            # Save the new best metric to a text file
            with open('best_metric.txt', 'w') as f:
                f.write(f'Best Validation Loss: {best_val_loss:.4f}')
            
            print(f"Saved new best metric: {best_val_loss:.4f} to best_metric.txt")
        
        # Early stopping check (assuming early_stopping is a callable object)
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break  # Break out of the loop if early stopping is triggered


    print('Finished Training')



def evaluate_on_test_set(model, test_loader, criteria, device):
    model.eval()
    all_labels = []
    all_predictions = []
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criteria(outputs.squeeze(), labels.float())
            test_loss += loss.item()
            predictions = outputs.squeeze().round()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
        
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    print(f'Test Loss: {test_loss / len(test_loader):.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    
        # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(all_labels, all_predictions)
    roc_auc = auc(fpr, tpr)
    print(f'AUC: {roc_auc:.4f}')
        
# Save confusion matrix as an image
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion Matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig('static/confusion_matrix.png')
    plt.close()

    # Save ROC curve as an image
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('static/roc_curve.png')
    plt.close()


    metrics = {
        'Test Loss': test_loss / len(test_loader),
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': roc_auc
        }
    return metrics

if __name__ == "__main__":
    train_model(model, train_loader, val_loader, criteria, optimizer, scheduler, device, num_epochs)
    # Optionally evaluate on the test set
    test_now = input("Evaluate on test set now? (y/n): ")
    if test_now.lower() == 'y':
        evaluate_on_test_set(model, test_loader, criteria, device)