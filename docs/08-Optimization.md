[Learn the Basics](intro.html) ||
[Quickstart](quickstart_tutorial.html) ||
[Tensors](tensorqs_tutorial.html) ||
[Datasets & DataLoaders](data_tutorial.html) ||
[Transforms](transforms_tutorial.html) ||
[Build Model](buildmodel_tutorial.html) ||
[Autograd](autogradqs_tutorial.html) ||
**Optimization** ||
[Save & Load Model](saveloadrun_tutorial.html)

# Optimizing Model Parameters

Now that we have a model and data it's time to train, validate and test our model by optimizing its parameters on
our data. Training a model is an iterative process; in each iteration the model makes a guess about the output, calculates
the error in its guess (*loss*), collects the derivatives of the error with respect to its parameters (as we saw in
the [previous section](autograd_tutorial.html)), and **optimizes** these parameters using gradient descent. For a more
detailed walkthrough of this process, check out this video on [backpropagation from 3Blue1Brown](https://www.youtube.com/watch?v=tIeHLnjs5U8)_.

## Prerequisite Code
We load the code from the previous sections on [Datasets & DataLoaders](data_tutorial.html)
and [Build Model](buildmodel_tutorial.html).



```python
%matplotlib inline

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
```

## Hyperparameters

Hyperparameters are adjustable parameters that let you control the model optimization process.
Different hyperparameter values can impact model training and convergence rates
([read more](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html)_ about hyperparameter tuning)

We define the following hyperparameters for training:
 - **Number of Epochs** - the number times to iterate over the dataset
 - **Batch Size** - the number of data samples propagated through the network before the parameters are updated
 - **Learning Rate** - how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training.





```python
learning_rate = 1e-3
batch_size = 64
epochs = 5
```

## Optimization Loop

Once we set our hyperparameters, we can then train and optimize our model with an optimization loop. Each
iteration of the optimization loop is called an **epoch**.

Each epoch consists of two main parts:
 - **The Train Loop** - iterate over the training dataset and try to converge to optimal parameters.
 - **The Validation/Test Loop** - iterate over the test dataset to check if model performance is improving.

Let's briefly familiarize ourselves with some of the concepts used in the training loop. Jump ahead to
see the `full-impl-label` of the optimization loop.

### Loss Function

When presented with some training data, our untrained network is likely not to give the correct
answer. **Loss function** measures the degree of dissimilarity of obtained result to the target value,
and it is the loss function that we want to minimize during training. To calculate the loss we make a
prediction using the inputs of our given data sample and compare it against the true data label value.

Common loss functions include [nn.MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss) (Mean Square Error) for regression tasks, and
[nn.NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss) (Negative Log Likelihood) for classification.
[nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss) combines ``nn.LogSoftmax`` and ``nn.NLLLoss``.

We pass our model's output logits to ``nn.CrossEntropyLoss``, which will normalize the logits and compute the prediction error.




```python
# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
```

### Optimizer

Optimization is the process of adjusting model parameters to reduce model error in each training step. **Optimization algorithms** define how this process is performed (in this example we use Stochastic Gradient Descent).
All optimization logic is encapsulated in  the ``optimizer`` object. Here, we use the SGD optimizer; additionally, there are many [different optimizers](https://pytorch.org/docs/stable/optim.html)
available in PyTorch such as ADAM and RMSProp, that work better for different kinds of models and data.

We initialize the optimizer by registering the model's parameters that need to be trained, and passing in the learning rate hyperparameter.




```python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

Inside the training loop, optimization happens in three steps:
 * Call ``optimizer.zero_grad()`` to reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.
 * Backpropagate the prediction loss with a call to ``loss.backward()``. PyTorch deposits the gradients of the loss w.r.t. each parameter.
 * Once we have our gradients, we call ``optimizer.step()`` to adjust the parameters by the gradients collected in the backward pass.




## Full Implementation
We define ``train_loop`` that loops over our optimization code, and ``test_loop`` that
evaluates the model's performance against our test data.




```python
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

We initialize the loss function and optimizer, and pass it to ``train_loop`` and ``test_loop``.
Feel free to increase the number of epochs to track the model's improving performance.




```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
```

    Epoch 1
    -------------------------------
    loss: 2.299604  [   64/60000]
    loss: 2.281797  [ 6464/60000]
    loss: 2.269583  [12864/60000]
    loss: 2.255457  [19264/60000]
    loss: 2.240205  [25664/60000]
    loss: 2.213762  [32064/60000]
    loss: 2.215705  [38464/60000]
    loss: 2.184422  [44864/60000]
    loss: 2.175044  [51264/60000]
    loss: 2.137501  [57664/60000]
    Test Error: 
     Accuracy: 53.1%, Avg loss: 2.138075 
    
    Epoch 2
    -------------------------------
    loss: 2.153558  [   64/60000]
    loss: 2.139259  [ 6464/60000]
    loss: 2.081727  [12864/60000]
    loss: 2.085114  [19264/60000]
    loss: 2.046907  [25664/60000]
    loss: 1.977491  [32064/60000]
    loss: 2.007782  [38464/60000]
    loss: 1.928677  [44864/60000]
    loss: 1.934681  [51264/60000]
    loss: 1.844566  [57664/60000]
    Test Error: 
     Accuracy: 59.0%, Avg loss: 1.855136 
    
    Epoch 3
    -------------------------------
    loss: 1.898872  [   64/60000]
    loss: 1.859855  [ 6464/60000]
    loss: 1.745800  [12864/60000]
    loss: 1.771856  [19264/60000]
    loss: 1.671929  [25664/60000]
    loss: 1.624660  [32064/60000]
    loss: 1.646571  [38464/60000]
    loss: 1.553838  [44864/60000]
    loss: 1.585847  [51264/60000]
    loss: 1.463247  [57664/60000]
    Test Error: 
     Accuracy: 62.0%, Avg loss: 1.491152 
    
    Epoch 4
    -------------------------------
    loss: 1.570554  [   64/60000]
    loss: 1.524995  [ 6464/60000]
    loss: 1.381242  [12864/60000]
    loss: 1.440385  [19264/60000]
    loss: 1.325888  [25664/60000]
    loss: 1.331313  [32064/60000]
    loss: 1.343411  [38464/60000]
    loss: 1.273921  [44864/60000]
    loss: 1.314914  [51264/60000]
    loss: 1.204072  [57664/60000]
    Test Error: 
     Accuracy: 63.9%, Avg loss: 1.234092 
    
    Epoch 5
    -------------------------------
    loss: 1.318503  [   64/60000]
    loss: 1.292388  [ 6464/60000]
    loss: 1.131896  [12864/60000]
    loss: 1.229624  [19264/60000]
    loss: 1.102847  [25664/60000]
    loss: 1.138407  [32064/60000]
    loss: 1.157674  [38464/60000]
    loss: 1.099932  [44864/60000]
    loss: 1.145054  [51264/60000]
    loss: 1.048841  [57664/60000]
    Test Error: 
     Accuracy: 65.2%, Avg loss: 1.074347 
    
    Epoch 6
    -------------------------------
    loss: 1.147973  [   64/60000]
    loss: 1.144627  [ 6464/60000]
    loss: 0.967731  [12864/60000]
    loss: 1.098405  [19264/60000]
    loss: 0.965783  [25664/60000]
    loss: 1.007831  [32064/60000]
    loss: 1.040992  [38464/60000]
    loss: 0.989532  [44864/60000]
    loss: 1.033878  [51264/60000]
    loss: 0.949742  [57664/60000]
    Test Error: 
     Accuracy: 66.5%, Avg loss: 0.970729 
    
    Epoch 7
    -------------------------------
    loss: 1.027588  [   64/60000]
    loss: 1.047764  [ 6464/60000]
    loss: 0.855220  [12864/60000]
    loss: 1.011105  [19264/60000]
    loss: 0.879051  [25664/60000]
    loss: 0.915307  [32064/60000]
    loss: 0.963445  [38464/60000]
    loss: 0.917342  [44864/60000]
    loss: 0.956093  [51264/60000]
    loss: 0.882487  [57664/60000]
    Test Error: 
     Accuracy: 67.8%, Avg loss: 0.899503 
    
    Epoch 8
    -------------------------------
    loss: 0.938215  [   64/60000]
    loss: 0.979911  [ 6464/60000]
    loss: 0.774328  [12864/60000]
    loss: 0.949241  [19264/60000]
    loss: 0.821273  [25664/60000]
    loss: 0.847455  [32064/60000]
    loss: 0.908044  [38464/60000]
    loss: 0.868443  [44864/60000]
    loss: 0.899046  [51264/60000]
    loss: 0.833970  [57664/60000]
    Test Error: 
     Accuracy: 69.0%, Avg loss: 0.847812 
    
    Epoch 9
    -------------------------------
    loss: 0.869330  [   64/60000]
    loss: 0.928490  [ 6464/60000]
    loss: 0.713798  [12864/60000]
    loss: 0.903194  [19264/60000]
    loss: 0.780315  [25664/60000]
    loss: 0.796318  [32064/60000]
    loss: 0.865808  [38464/60000]
    loss: 0.834232  [44864/60000]
    loss: 0.856279  [51264/60000]
    loss: 0.797004  [57664/60000]
    Test Error: 
     Accuracy: 70.2%, Avg loss: 0.808560 
    
    Epoch 10
    -------------------------------
    loss: 0.814243  [   64/60000]
    loss: 0.887317  [ 6464/60000]
    loss: 0.666916  [12864/60000]
    loss: 0.867729  [19264/60000]
    loss: 0.749378  [25664/60000]
    loss: 0.757221  [32064/60000]
    loss: 0.831676  [38464/60000]
    loss: 0.808831  [44864/60000]
    loss: 0.822820  [51264/60000]
    loss: 0.767592  [57664/60000]
    Test Error: 
     Accuracy: 71.5%, Avg loss: 0.777352 
    
    Done!


## Further Reading
- [Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
- [torch.optim](https://pytorch.org/docs/stable/optim.html)
- [Warmstart Training a Model](https://pytorch.org/tutorials/recipes/recipes/warmstarting_model_using_parameters_from_a_different_model.html)



