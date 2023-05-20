import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class LeNet(nn.Module):
    def __init__(self) -> None:
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    

def train(model, device, train_loader, test_loader, optimizer, epoch):
    for ep in range(epoch):
        print(f"Epoch {ep} is training =============================")
        model.train()
        for data, target in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        print(f"Epoch {ep} is evaluating =============================")
        eval_in_training(model, device, test_loader)


def eval_in_training(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduce='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:,.0f}%)\n".format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    use_cuda = True
    device = torch.device("cuda")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = datasets.MNIST('data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('data', train=False, download=True, transform=transform)
    
    batch_size = 32
    lr = 0.1
    epoch = 5
    model = LeNet().to(device=device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    train_loader = torch.utils.data.DataLoader(dataset1, pin_memory=True, shuffle=True, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(dataset2, pin_memory=True, shuffle=True, batch_size=1)

    train(model=model, device=device, test_loader=test_loader, train_loader=train_loader, optimizer=optimizer, epoch=epoch)

    torch.save(model.state_dict(), "mnist_cnn.pt")