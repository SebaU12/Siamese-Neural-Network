import torch
import torch.nn as nn
from torchvision import transforms
from dataset.WildFacesDataset import buildTrainTestDataset
from pytorch_model_summary import summary
import matplotlib.pyplot as plt
from snn import SiameseNeuralNetwork, SingleUnitNetwork

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

learning_rate = 0.0001
num_epochs = 30
batch_size = 32
num_channels = 1

transform = transforms.Compose([
    transforms.Resize((250, 250)),
    transforms.CenterCrop(100),
    transforms.Grayscale(num_output_channels=num_channels),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


torch.cuda.empty_cache()

train_dataloader, test_dataloader =  buildTrainTestDataset(batch_size=batch_size, 
                                                           csv_file='./train.csv', 
                                                           img_dir='./images/images', 
                                                           img_ext='png',
                                                           transform=transform)

    
model = SiameseNeuralNetwork().to(device)

def train(model, train_loader, val_loader, num_epochs, learning_rate):
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    total_step = len(train_loader)
    list_loss = []
    list_time = []

    for epoch in range(num_epochs):
        i = 0
        model.train()
        epoch_train_loss = 0
        correct_train = 0
        total_train = 0

        for img1, img2, labels in train_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.float().to(device)

            # Forward pass
            output = model(img1, img2).squeeze()
            loss = loss_fn(output, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            list_loss.append(loss.item())
            list_time.append(i)
            i += 1
            predicted = (torch.sigmoid(output) > 0.5).float()
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            if (i + 1) % 5 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        val_loss, val_accuracy = validate(model, val_loader, loss_fn)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

    print('Finished Training')
    return train_losses, val_losses, train_accuracies, val_accuracies

# Función para validar el modelo
def validate(model, val_loader, loss_fn):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for img1, img2, labels in val_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            output = model(img1, img2).squeeze()  # Asegurarse de que la salida tenga la misma dimensión que las etiquetas
            loss = loss_fn(output, labels.float())  # Convertir las etiquetas a tipo float
            val_loss += loss.item()

            predicted = (output > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    return avg_val_loss, val_accuracy

# Función para graficar las métricas
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Train Loss')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b', label='Train Accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.legend()

    plt.savefig('Train_Validation_Loss.png')


print(summary(SingleUnitNetwork(), torch.zeros((batch_size, num_channels, 100, 100)), show_input=True))
print(summary(SiameseNeuralNetwork(), torch.zeros((batch_size, num_channels, 100, 100)), torch.zeros((batch_size, num_channels, 100, 100)), show_input=True))


print("Iniciando Fase entrenamiento")
train_losses, val_losses, train_accuracies, val_accuracies = train(model, train_dataloader, test_dataloader, num_epochs, learning_rate)
print("Generando grafica de metrica")
plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

print("Guardando modelo")
torch.save(model.state_dict(), 'modelo.pth')