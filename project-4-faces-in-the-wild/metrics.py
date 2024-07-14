import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from dataset.WildFacesDataset import buildTrainTestDataset

from snn import SiameseNeuralNetwork
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score, f1_score


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

model = SiameseNeuralNetwork().to(device)
model.load_state_dict(torch.load('modelo.pth'))
model.eval()

transform = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.CenterCrop(100),
        transforms.Grayscale(num_output_channels=1),
        #transforms.RandomVerticalFlip(p=0.5),
        #transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataloader, test_dataloader =  buildTrainTestDataset(batch_size=1, 
                                                               csv_file='./train.csv', 
                                                               img_dir='./images/images', 
                                                               img_ext='png',
                                                               transform=transform)

true_labels = []
predicted_labels = []
with torch.no_grad():
    for img1, img2, label in test_dataloader:
        img1, img2 = img1.to(device), img2.to(device)
        output = model(img1, img2)
        predicted = (torch.sigmoid(output) > 0.5).float()
        predicted = predicted.cpu().numpy()
        true_labels.append(label.numpy()[0])
        predicted_labels.append(predicted[0][0])



cm = confusion_matrix(true_labels, predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Same', 'Same'])

# Guardar la matriz de confusión como imagen
fig, ax = plt.subplots(figsize=(15, 15))
disp.plot(cmap='viridis', ax=ax, values_format='d')

# Aumentar el tamaño de los números dentro de la matriz de confusión
for text in disp.text_.ravel():
    text.set_fontsize(20)

plt.title('Confusion Matrix for Siamese Neural Network', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Predicted Label', fontsize=18)
plt.ylabel('True Label', fontsize=18)
plt.savefig('confusion_matrix.png', bbox_inches='tight')
plt.show()
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
accuracy = accuracy_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

print(f"Precisión: {precision}")
print(f"Recall: {recall}")
print(f"Accuracy: {accuracy}")
print(f"F1-score: {f1}")