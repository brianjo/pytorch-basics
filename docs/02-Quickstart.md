[Learn the Basics](Introduction.html) ||
**Quickstart** ||
[Tensors](Tensors.html) ||
[Datasets & DataLoaders](Data.html) ||
[Transforms](transforms_tutorial.html) ||
[Build Model](buildmodel_tutorial.html) ||
[Autograd](autogradqs_tutorial.html) ||
[Optimization](optimization_tutorial.html) ||
[Save & Load Model](saveloadrun_tutorial.html)

# Quickstart
This section runs through the API for common tasks in machine learning. Refer to the links in each section to dive deeper.

## Working with data
PyTorch has two [primitives to work with data](https://pytorch.org/docs/stable/data.html):
``torch.utils.data.DataLoader`` and ``torch.utils.data.Dataset``.
``Dataset`` stores the samples and their corresponding labels, and ``DataLoader`` wraps an iterable around
the ``Dataset``.



```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
```

PyTorch offers domain-specific libraries such as [TorchText](https://pytorch.org/text/stable/index.html),
[TorchVision](https://pytorch.org/vision/stable/index.html), and [TorchAudio](https://pytorch.org/audio/stable/index.html),
all of which include datasets. For this tutorial, we  will be using a TorchVision dataset.

The ``torchvision.datasets`` module contains ``Dataset`` objects for many real-world vision data like
CIFAR, COCO ([full list here](https://pytorch.org/vision/stable/datasets.html)). In this tutorial, we
use the FashionMNIST dataset. Every TorchVision ``Dataset`` includes two arguments: ``transform`` and
``target_transform`` to modify the samples and labels respectively.




```python
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
```

We pass the ``Dataset`` as an argument to ``DataLoader``. This wraps an iterable over our dataset, and supports
automatic batching, sampling, shuffling and multiprocess data loading. Here we define a batch size of 64, i.e. each element
in the dataloader iterable will return a batch of 64 features and labels.




```python
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
```

    Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
    Shape of y: torch.Size([64]) torch.int64


Read more about [loading data in PyTorch](data_tutorial.html).




--------------




## Creating Models
To define a neural network in PyTorch, we create a class that inherits
from [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html). We define the layers of the network
in the ``__init__`` function and specify how data will pass through the network in the ``forward`` function. To accelerate
operations in the neural network, we move it to the GPU if available.




```python
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
```

    Using mps device
    NeuralNetwork(
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (linear_relu_stack): Sequential(
        (0): Linear(in_features=784, out_features=512, bias=True)
        (1): ReLU()
        (2): Linear(in_features=512, out_features=512, bias=True)
        (3): ReLU()
        (4): Linear(in_features=512, out_features=10, bias=True)
      )
    )


Read more about [building neural networks in PyTorch](buildmodel_tutorial.html).




--------------




## Optimizing the Model Parameters
To train a model, we need a [loss function](https://pytorch.org/docs/stable/nn.html#loss-functions)
and an [optimizer](https://pytorch.org/docs/stable/optim.html).




```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```

In a single training loop, the model makes predictions on the training dataset (fed to it in batches), and
backpropagates the prediction error to adjust the model's parameters.




```python
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

We also check the model's performance against the test dataset to ensure it is learning.




```python
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

The training process is conducted over several iterations (*epochs*). During each epoch, the model learns
parameters to make better predictions. We print the model's accuracy and loss at each epoch; we'd like to see the
accuracy increase and the loss decrease with every epoch.




```python
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
```

    Epoch 1
    -------------------------------
    loss: 2.300994  [    0/60000]
    loss: 2.289627  [ 6400/60000]
    loss: 2.278757  [12800/60000]
    loss: 2.273481  [19200/60000]
    loss: 2.260533  [25600/60000]
    loss: 2.230715  [32000/60000]
    loss: 2.240870  [38400/60000]
    loss: 2.210235  [44800/60000]
    loss: 2.205794  [51200/60000]
    loss: 2.179301  [57600/60000]
    Test Error: 
     Accuracy: 42.7%, Avg loss: 2.175595 
    
    Epoch 2
    -------------------------------
    loss: 2.179688  [    0/60000]
    loss: 2.170581  [ 6400/60000]
    loss: 2.125383  [12800/60000]
    loss: 2.134987  [19200/60000]
    loss: 2.104071  [25600/60000]
    loss: 2.039638  [32000/60000]
    loss: 2.065766  [38400/60000]
    loss: 1.994649  [44800/60000]
    loss: 1.991123  [51200/60000]
    loss: 1.927214  [57600/60000]
    Test Error: 
     Accuracy: 56.1%, Avg loss: 1.929943 
    
    Epoch 3
    -------------------------------
    loss: 1.957387  [    0/60000]
    loss: 1.929036  [ 6400/60000]
    loss: 1.825893  [12800/60000]
    loss: 1.850506  [19200/60000]
    loss: 1.775094  [25600/60000]
    loss: 1.708617  [32000/60000]
    loss: 1.727947  [38400/60000]
    loss: 1.628896  [44800/60000]
    loss: 1.653404  [51200/60000]
    loss: 1.548985  [57600/60000]
    Test Error: 
     Accuracy: 60.7%, Avg loss: 1.570322 
    
    Epoch 4
    -------------------------------
    loss: 1.634544  [    0/60000]
    loss: 1.598077  [ 6400/60000]
    loss: 1.457816  [12800/60000]
    loss: 1.511364  [19200/60000]
    loss: 1.425202  [25600/60000]
    loss: 1.398494  [32000/60000]
    loss: 1.412483  [38400/60000]
    loss: 1.328141  [44800/60000]
    loss: 1.371268  [51200/60000]
    loss: 1.270080  [57600/60000]
    Test Error: 
     Accuracy: 63.2%, Avg loss: 1.298073 
    
    Epoch 5
    -------------------------------
    loss: 1.375485  [    0/60000]
    loss: 1.353134  [ 6400/60000]
    loss: 1.197045  [12800/60000]
    loss: 1.282228  [19200/60000]
    loss: 1.185837  [25600/60000]
    loss: 1.195442  [32000/60000]
    loss: 1.213788  [38400/60000]
    loss: 1.140980  [44800/60000]
    loss: 1.188507  [51200/60000]
    loss: 1.102179  [57600/60000]
    Test Error: 
     Accuracy: 64.6%, Avg loss: 1.124997 
    
    Done!


Read more about [Training your model](optimization_tutorial.html).




--------------




## Saving Models
A common way to save a model is to serialize the internal state dictionary (containing the model parameters).




```python
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
```

    Saved PyTorch Model State to model.pth


## Loading Models

The process for loading a model includes re-creating the model structure and loading
the state dictionary into it.




```python
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))
```




    <All keys matched successfully>



This model can now be used to make predictions.




```python
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
```

    Predicted: "Ankle boot", Actual: "Ankle boot"


Read more about [Saving & Loading your model](saveloadrun_tutorial.html).



