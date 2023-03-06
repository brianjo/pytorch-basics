```python
%matplotlib inline
```


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
    loss: 2.310308  [   64/60000]
    loss: 2.291682  [ 6464/60000]
    loss: 2.282847  [12864/60000]
    loss: 2.278148  [19264/60000]
    loss: 2.259573  [25664/60000]
    loss: 2.246842  [32064/60000]
    loss: 2.237948  [38464/60000]
    loss: 2.221490  [44864/60000]
    loss: 2.215676  [51264/60000]
    loss: 2.186174  [57664/60000]
    Test Error: 
     Accuracy: 50.1%, Avg loss: 2.185173 
    
    Epoch 2
    -------------------------------
    loss: 2.192464  [   64/60000]
    loss: 2.176265  [ 6464/60000]
    loss: 2.138019  [12864/60000]
    loss: 2.155484  [19264/60000]
    loss: 2.096774  [25664/60000]
    loss: 2.064352  [32064/60000]
    loss: 2.073422  [38464/60000]
    loss: 2.019561  [44864/60000]
    loss: 2.018754  [51264/60000]
    loss: 1.944076  [57664/60000]
    Test Error: 
     Accuracy: 56.9%, Avg loss: 1.951974 
    
    Epoch 3
    -------------------------------
    loss: 1.979550  [   64/60000]
    loss: 1.944613  [ 6464/60000]
    loss: 1.850896  [12864/60000]
    loss: 1.885921  [19264/60000]
    loss: 1.766024  [25664/60000]
    loss: 1.721881  [32064/60000]
    loss: 1.732149  [38464/60000]
    loss: 1.646069  [44864/60000]
    loss: 1.663508  [51264/60000]
    loss: 1.542335  [57664/60000]
    Test Error: 
     Accuracy: 60.8%, Avg loss: 1.575167 
    
    Epoch 4
    -------------------------------
    loss: 1.641383  [   64/60000]
    loss: 1.597785  [ 6464/60000]
    loss: 1.460881  [12864/60000]
    loss: 1.522893  [19264/60000]
    loss: 1.394849  [25664/60000]
    loss: 1.381750  [32064/60000]
    loss: 1.389999  [38464/60000]
    loss: 1.324359  [44864/60000]
    loss: 1.359623  [51264/60000]
    loss: 1.242349  [57664/60000]
    Test Error: 
     Accuracy: 63.2%, Avg loss: 1.281596 
    
    Epoch 5
    -------------------------------
    loss: 1.364956  [   64/60000]
    loss: 1.337699  [ 6464/60000]
    loss: 1.179997  [12864/60000]
    loss: 1.276043  [19264/60000]
    loss: 1.145318  [25664/60000]
    loss: 1.163051  [32064/60000]
    loss: 1.179221  [38464/60000]
    loss: 1.127842  [44864/60000]
    loss: 1.170320  [51264/60000]
    loss: 1.072596  [57664/60000]
    Test Error: 
     Accuracy: 64.8%, Avg loss: 1.102368 
    
    Epoch 6
    -------------------------------
    loss: 1.181124  [   64/60000]
    loss: 1.175671  [ 6464/60000]
    loss: 0.999543  [12864/60000]
    loss: 1.125861  [19264/60000]
    loss: 0.994338  [25664/60000]
    loss: 1.020635  [32064/60000]
    loss: 1.052101  [38464/60000]
    loss: 1.005876  [44864/60000]
    loss: 1.050259  [51264/60000]
    loss: 0.969423  [57664/60000]
    Test Error: 
     Accuracy: 65.8%, Avg loss: 0.989962 
    
    Epoch 7
    -------------------------------
    loss: 1.055653  [   64/60000]
    loss: 1.073796  [ 6464/60000]
    loss: 0.878792  [12864/60000]
    loss: 1.027988  [19264/60000]
    loss: 0.902191  [25664/60000]
    loss: 0.923560  [32064/60000]
    loss: 0.970771  [38464/60000]
    loss: 0.927402  [44864/60000]
    loss: 0.969056  [51264/60000]
    loss: 0.901827  [57664/60000]
    Test Error: 
     Accuracy: 66.8%, Avg loss: 0.914991 
    
    Epoch 8
    -------------------------------
    loss: 0.964512  [   64/60000]
    loss: 1.004631  [ 6464/60000]
    loss: 0.793878  [12864/60000]
    loss: 0.959500  [19264/60000]
    loss: 0.842306  [25664/60000]
    loss: 0.854395  [32064/60000]
    loss: 0.914801  [38464/60000]
    loss: 0.875149  [44864/60000]
    loss: 0.910963  [51264/60000]
    loss: 0.853945  [57664/60000]
    Test Error: 
     Accuracy: 67.8%, Avg loss: 0.861828 
    
    Epoch 9
    -------------------------------
    loss: 0.895530  [   64/60000]
    loss: 0.953656  [ 6464/60000]
    loss: 0.731293  [12864/60000]
    loss: 0.908750  [19264/60000]
    loss: 0.800252  [25664/60000]
    loss: 0.803487  [32064/60000]
    loss: 0.873069  [38464/60000]
    loss: 0.838708  [44864/60000]
    loss: 0.867891  [51264/60000]
    loss: 0.817475  [57664/60000]
    Test Error: 
     Accuracy: 68.9%, Avg loss: 0.821918 
    
    Epoch 10
    -------------------------------
    loss: 0.841097  [   64/60000]
    loss: 0.913210  [ 6464/60000]
    loss: 0.683007  [12864/60000]
    loss: 0.869649  [19264/60000]
    loss: 0.768555  [25664/60000]
    loss: 0.764901  [32064/60000]
    loss: 0.839639  [38464/60000]
    loss: 0.811697  [44864/60000]
    loss: 0.834432  [51264/60000]
    loss: 0.788075  [57664/60000]
    Test Error: 
     Accuracy: 70.1%, Avg loss: 0.790321 
    
    Done!


## Further Reading
- [Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
- [torch.optim](https://pytorch.org/docs/stable/optim.html)
- [Warmstart Training a Model](https://pytorch.org/tutorials/recipes/recipes/warmstarting_model_using_parameters_from_a_different_model.html)



