import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torchvision

from model import Pose3D, Extractor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trainExtractor(trainloader, epochs=3, lr=0.01, momentum=0.9, weight_decay=0.0001):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(extractor.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

    for epoch in range(epochs):
        losses = []
        running_loss = 0
        for i, inp in enumerate(trainloader):
            inputs, labels = inp
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = extractor(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 1 == 0 and i > 0:
                print(f'Loss [{epoch+1}, {i}](epoch, minibatch): ', running_loss / 1)
                running_loss = 0.0

        avg_loss = sum(losses) / len(losses)
        scheduler.step(avg_loss)


def testExtractor(testloader):
    criterion = nn.MSELoss()
    losses = []

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = extractor(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
        avg_loss = sum(losses) / len(losses)
        print("Avg Loss: ", avg_loss)


def train3DPose():
    def loss_func(P, P_, HM, HM_):
        lamda_P = 1e-1
        loss_P = 0

        lamda_phi = -1e-2
        loss_phi = 0
        lamda_L = 0.5
        loss_L = 0
        loss_R = lamda_phi * loss_phi + lamda_L * loss_L

        lamda_HM = 1e-3
        loss_HM = 0

        return lamda_P * loss_P + loss_R + lamda_HM * loss_HM

    pass


def test3DPose():
    pass


if __name__ == "__main__":
    extractor = pose_model.extractor.to(device)
    # trainExtractor
    pose_model = Pose3D(extractor=extractor).to(device)
    # train3DPose
