import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)  # (B, 8, 14, 14)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1) # (B, 16, 7, 7)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) # (B, 32, 4, 4)
        self.fc1 = nn.Linear(32*4*4, 32)
        self.fc2 = nn.Linear(32, 12)  # Latent space of dimension 12

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
        self.fc1 = nn.Linear(12, 32)
        self.fc2 = nn.Linear(32, 32*4*4)
        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1)
        # todo is it ok that padding=3 here??
        self.deconv3 = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=3, output_padding=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = x.view(x.size(0), 32, 4, 4)
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


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.encoder = Encoder()
        self.lay1 = nn.Linear(12, 10)
        self.lay2 = nn.Linear(10, 10)

        # self.lay3 = nn.Linear(10, 1)

    def forward(self, x):
        latent = self.encoder(x)
        x = torch.relu(self.lay1(latent))

        x = torch.sigmoid(self.lay2(x))

        # x = torch.relu(self.lay2(x))
        # x = torch.sigmoid(self.lay3(x))
        return x


def train_autoencoder(cur_autoencoder, cur_criterion, cur_optimizer, cur_train_loader, cur_test_loader, plot: bool = True):
    train_losses = []
    test_losses = []

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        for data in cur_train_loader:
            inputs, _ = data
            inputs = inputs.to(device)

            cur_optimizer.zero_grad()
            outputs = cur_autoencoder(inputs)
            loss = cur_criterion(outputs, inputs)
            loss.backward()

            cur_optimizer.step()

        for data in cur_test_loader:
            inputs, _ = data
            inputs = inputs.to(device)
            outputs = cur_autoencoder(inputs)
            test_loss = cur_criterion(outputs, inputs)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

        if plot:
            train_losses.append(loss.item())
            test_losses.append(test_loss.item())

    # Evaluate the autoencoder
    cur_autoencoder.eval()
    # dataiter = iter(test_loader)
    # images, labels = dataiter.next()
    if plot:
        # Create an array of iterations
        iterations = list(range(len(train_losses)))

        # Plotting
        plt.plot(iterations, train_losses, marker='o', label='Train Loss')
        plt.plot(iterations, test_losses, marker='x', label='Test Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss vs Iteration')
        plt.grid(True)
        plt.show()


def train_classifier(cur_classifier, cur_criterion, cur_optimizer, cur_train_loader, cur_test_loader, plot: bool = True):
    train_losses = []
    test_losses = []

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        for data in cur_train_loader:
            inputs, true_outputs = data
            inputs, true_outputs = inputs.to(device), true_outputs.to(device)

            cur_optimizer.zero_grad()
            outputs = cur_classifier(inputs)

            true_outputs = torch.nn.functional.one_hot(true_outputs, num_classes=10)
            true_outputs = true_outputs.float()

            loss = cur_criterion(outputs, true_outputs)
            loss.backward()
            cur_optimizer.step()

        for data in cur_test_loader:
            inputs, true_outputs = data
            inputs, true_outputs = inputs.to(device), true_outputs.to(device)

            true_outputs = torch.nn.functional.one_hot(true_outputs, num_classes=10)
            true_outputs = true_outputs.float()

            outputs = cur_classifier(inputs)
            test_loss = cur_criterion(outputs, true_outputs)

        if plot:
            train_losses.append(loss.item())
            test_losses.append(test_loss.item())

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

    if plot:
        # Create an array of iterations
        iterations = list(range(len(train_losses)))

        # Plotting
        plt.plot(iterations, train_losses, marker='o', label='Train Loss')
        plt.plot(iterations, test_losses, marker='x', label='Test Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss vs Iteration')
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Q1 - Autoencoder
    # print("Training Autoencoder")
    #
    # # Define the autoencoder, loss function, and optimizer
    # autoencoder = Autoencoder()
    # autoencoder.to(device)
    #
    # criterion = nn.L1Loss()
    # optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    #
    # train_autoencoder(autoencoder, criterion, optimizer, train_loader, test_loader)

    # Q2 - Classifier
    print("Training Classifier")

    classifier = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    train_classifier(classifier, criterion, optimizer, train_loader, test_loader)

    # Q3 - Decoding
    print("Training Decoder")

    for param in classifier.encoder.parameters():
        param.requires_grad = False

    second_autoencoder = Autoencoder().to(device)
    second_autoencoder.encoder = classifier.encoder
    criterion = nn.L1Loss()
    optimizer = optim.Adam(second_autoencoder.parameters(), lr=0.001)

    train_autoencoder(second_autoencoder, criterion, optimizer, train_loader, test_loader)

    # # Q4 - Too few Examples
    # print("Too few Examples")
    # indices = torch.arange(100)
    # train_loader_CLS = torch.utils.data.Subset(train_dataset, indices)
    # train_loader_CLS = torch.utils.data.DataLoader(train_loader_CLS,
    #                                                batch_size=10, shuffle=True, num_workers=0)
    #
    # classifier = Classifier()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    #
    # train_classifier(classifier, criterion, optimizer, train_loader_CLS)
    #
    # autoencoder = Autoencoder()
    # criterion = nn.L1Loss()
    # optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    #
    # train_autoencoder(autoencoder, criterion, optimizer, train_loader_CLS)
    #
    # # Q5 - Transfer Learning
    # print("Transfer Learning")

