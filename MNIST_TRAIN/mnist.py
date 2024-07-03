"""
MNIST with PyTorch 

Script will be linked in the description as a Github Gist.

Install PyTorch nightly with this command:
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
"""

import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
#from  MNIST_MODEL.Net import Net

import sys, getopt
from  MNIST_MODEL.Net import Net 


def train(model, device, train_loader, optimizer, epoch, PATH):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                 100. * batch_idx / len(train_loader), loss.item()))
    # Save the model
    print('SAVE MODEL TO ---> ', PATH)
    torch.save(model.state_dict(), PATH, _use_new_zipfile_serialization=False)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main(argv):
    EPOCHS = 0
    MODEL_PATH  = ''
    try:
      opts, args = getopt.getopt(argv,"he:p:",["epochs=","path="])
    except getopt.GetoptError:
      print ('mnist.py -e <NUMBER_OF_EPOCHS> -p <MODEL_PATH>')
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
         print ('test.py -e <NUMBER_OF_EPOCHS> -p <MODEL_PATH>')
         sys.exit()
      elif opt in ("-e", "--epochs"):
         EPOCHS = int(arg)
      elif opt in ("-p", "--path"):
         MODEL_PATH = arg
    print ('Number Of Epochs = "', EPOCHS)
    print ('Path OF Model =', MODEL_PATH)


    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Using Device: ", device)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=128, shuffle=True)


    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(1, EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch, MODEL_PATH)
        test(model, device, test_loader)

    

if __name__ == "__main__":
    main(sys.argv[1:])


