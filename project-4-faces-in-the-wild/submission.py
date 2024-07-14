import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from snn import SiameseNeuralNetwork
from dataset.SubmissionDataset import SubmissionDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.Compose([
    transforms.Resize((250, 250)),
    transforms.CenterCrop(100),
    transforms.Grayscale(num_output_channels=1),
    #transforms.RandomVerticalFlip(p=0.5),
    #transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

csv_file='./test.csv'
img_dir='./images/images'
img_ext='png'
test_dataset  = SubmissionDataset(csv_file=csv_file, img_dir=img_dir, img_ext=img_ext, transform=transform)
dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = SiameseNeuralNetwork().to(device)
model.load_state_dict(torch.load('modelo.pth'))
model.eval()  # Establecer el modelo en modo evaluación
predictions = []
image_pairs = []
with torch.no_grad():
    for img1, img2, name in dataloader:
        img1, img2 = img1.to(device), img2.to(device)
        output = model(img1, img2)
        predicted = (output > 0.5).float() 
        predicted = predicted.cpu().numpy()
        predicted = predicted[0][0]
        if (predicted == 0):
            label = 'diff'
        else:
            label = 'same'
        image_pairs.append(name[0])
        predictions.append(label)

df = pd.DataFrame({
        'image1_image2': image_pairs,
        'label': predictions
    })

    # Guardar el DataFrame en un archivo CSV
df.to_csv('output.csv', index=False)