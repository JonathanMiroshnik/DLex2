import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)  # (B, 32, 14, 14)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # (B, 64, 7, 7)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # (B, 128, 4, 4)
        self.fc1 = nn.Linear(128*4*4, 256)
        self.fc2 = nn.Linear(256, 12)  # Latent space of dimension 12

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(12, 256)
        self.fc2 = nn.Linear(256, 128*4*4)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = x.view(x.size(0), 128, 4, 4)
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))  # Sigmoid activation for final layer to match the input range [0, 1]
        return x



class FC_Layer(nn.Module):
    def __init__(self):
        super(FC_Layer, self).__init__()
        self.fc = nn.Linear(128 * 4 * 4, 12)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


# Function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Define the autoencoder, loss function, and optimizer
    autoencoder = Autoencoder()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        for data in train_loader:
            inputs, _ = data
            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluate the autoencoder
    autoencoder.eval()
    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    # Get reconstructed images
    reconstructed = autoencoder(images)

    # Show original images
    print('Original Images')
    imshow(torchvision.utils.make_grid(images))

    # Show reconstructed images
    print('Reconstructed Images')
    imshow(torchvision.utils.make_grid(reconstructed.detach()))

