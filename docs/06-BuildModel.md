[Learn the Basics](intro.html) ||
[Quickstart](quickstart_tutorial.html) ||
[Tensors](tensorqs_tutorial.html) ||
[Datasets & DataLoaders](data_tutorial.html) ||
[Transforms](transforms_tutorial.html) ||
**Build Model** ||
[Autograd](autogradqs_tutorial.html) ||
[Optimization](optimization_tutorial.html) ||
[Save & Load Model](saveloadrun_tutorial.html)

# Build the Neural Network

Neural networks comprise of layers/modules that perform operations on data.
The [torch.nn](https://pytorch.org/docs/stable/nn.html) namespace provides all the building blocks you need to
build your own neural network. Every module in PyTorch subclasses the [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html).
A neural network is a module itself that consists of other modules (layers). This nested structure allows for
building and managing complex architectures easily.

In the following sections, we'll build a neural network to classify images in the FashionMNIST dataset.



```python
%matplotlib inline

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
```

## Get Device for Training
We want to be able to train our model on a hardware accelerator like the GPU,
if it is available. Let's check to see if
[torch.cuda](https://pytorch.org/docs/stable/notes/cuda.html) is available, else we
continue to use the CPU.




```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
```

    Using cpu device


## Define the Class
We define our neural network by subclassing ``nn.Module``, and
initialize the neural network layers in ``__init__``. Every ``nn.Module`` subclass implements
the operations on input data in the ``forward`` method.




```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
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
```

We create an instance of ``NeuralNetwork``, and move it to the ``device``, and print
its structure.




```python
model = NeuralNetwork().to(device)
print(model)
```

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


To use the model, we pass it the input data. This executes the model's ``forward``,
along with some [background operations](https://github.com/pytorch/pytorch/blob/270111b7b611d174967ed204776985cefca9c144/torch/nn/modules/module.py#L866).
Do not call ``model.forward()`` directly!

Calling the model on the input returns a 2-dimensional tensor with dim=0 corresponding to each output of 10 raw predicted values for each class, and dim=1 corresponding to the individual values of each output.
We get the prediction probabilities by passing it through an instance of the ``nn.Softmax`` module.




```python
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
```

    Predicted class: tensor([9])


--------------




## Model Layers

Let's break down the layers in the FashionMNIST model. To illustrate it, we
will take a sample minibatch of 3 images of size 28x28 and see what happens to it as
we pass it through the network.




```python
input_image = torch.rand(3,28,28)
print(input_image.size())
```

    torch.Size([3, 28, 28])


### nn.Flatten
We initialize the [nn.Flatten](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html)
layer to convert each 2D 28x28 image into a contiguous array of 784 pixel values (
the minibatch dimension (at dim=0) is maintained).




```python
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())
```

    torch.Size([3, 784])


### nn.Linear
The [linear layer](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
is a module that applies a linear transformation on the input using its stored weights and biases.





```python
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())
```

    torch.Size([3, 20])


### nn.ReLU
Non-linear activations are what create the complex mappings between the model's inputs and outputs.
They are applied after linear transformations to introduce *nonlinearity*, helping neural networks
learn a wide variety of phenomena.

In this model, we use [nn.ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html) between our
linear layers, but there's other activations to introduce non-linearity in your model.




```python
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")
```

    Before ReLU: tensor([[-5.5712e-01,  4.1135e-01, -7.4510e-03, -5.4891e-02,  7.3538e-02,
              4.6617e-01,  5.3287e-01,  7.2283e-02, -3.7471e-01, -3.9285e-01,
             -6.7889e-01,  2.1088e-01,  1.8742e-01,  4.0150e-01, -5.6422e-02,
             -4.8977e-02, -1.6230e-01,  3.0556e-01, -7.1455e-01, -6.6180e-02],
            [-4.2601e-01,  6.2487e-01, -5.9415e-02,  2.3934e-02,  3.9810e-01,
              3.2441e-01,  7.0026e-01, -1.2423e-01, -5.2260e-01, -1.7234e-01,
             -5.5835e-01,  2.2128e-01,  2.7830e-01,  2.4191e-01, -7.7681e-02,
             -2.4954e-01,  1.5836e-01,  1.9990e-01, -1.1715e-01, -3.2138e-01],
            [-4.9225e-01,  4.1050e-01, -1.5492e-01,  8.9106e-03,  3.5985e-01,
              3.1355e-01,  6.2615e-01, -1.9053e-04, -5.7080e-01, -1.7064e-01,
             -6.5802e-01,  3.3700e-01,  4.5726e-01,  3.1022e-01, -4.0316e-01,
             -3.8029e-01, -1.2243e-01,  3.6732e-01, -5.6789e-01, -9.4490e-02]],
           grad_fn=<AddmmBackward0>)
    
    
    After ReLU: tensor([[0.0000, 0.4113, 0.0000, 0.0000, 0.0735, 0.4662, 0.5329, 0.0723, 0.0000,
             0.0000, 0.0000, 0.2109, 0.1874, 0.4015, 0.0000, 0.0000, 0.0000, 0.3056,
             0.0000, 0.0000],
            [0.0000, 0.6249, 0.0000, 0.0239, 0.3981, 0.3244, 0.7003, 0.0000, 0.0000,
             0.0000, 0.0000, 0.2213, 0.2783, 0.2419, 0.0000, 0.0000, 0.1584, 0.1999,
             0.0000, 0.0000],
            [0.0000, 0.4105, 0.0000, 0.0089, 0.3599, 0.3136, 0.6262, 0.0000, 0.0000,
             0.0000, 0.0000, 0.3370, 0.4573, 0.3102, 0.0000, 0.0000, 0.0000, 0.3673,
             0.0000, 0.0000]], grad_fn=<ReluBackward0>)


### nn.Sequential
[nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) is an ordered
container of modules. The data is passed through all the modules in the same order as defined. You can use
sequential containers to put together a quick network like ``seq_modules``.




```python
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)
```

### nn.Softmax
The last linear layer of the neural network returns `logits` - raw values in [-\infty, \infty] - which are passed to the
[nn.Softmax](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html) module. The logits are scaled to values
[0, 1] representing the model's predicted probabilities for each class. ``dim`` parameter indicates the dimension along
which the values must sum to 1.




```python
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
```

## Model Parameters
Many layers inside a neural network are *parameterized*, i.e. have associated weights
and biases that are optimized during training. Subclassing ``nn.Module`` automatically
tracks all fields defined inside your model object, and makes all parameters
accessible using your model's ``parameters()`` or ``named_parameters()`` methods.

In this example, we iterate over each parameter, and print its size and a preview of its values.





```python
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
```

    Model structure: NeuralNetwork(
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (linear_relu_stack): Sequential(
        (0): Linear(in_features=784, out_features=512, bias=True)
        (1): ReLU()
        (2): Linear(in_features=512, out_features=512, bias=True)
        (3): ReLU()
        (4): Linear(in_features=512, out_features=10, bias=True)
      )
    )
    
    
    Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values : tensor([[ 0.0211,  0.0168,  0.0334,  ..., -0.0151, -0.0033,  0.0032],
            [-0.0022,  0.0293, -0.0090,  ..., -0.0044, -0.0147, -0.0251]],
           grad_fn=<SliceBackward0>) 
    
    Layer: linear_relu_stack.0.bias | Size: torch.Size([512]) | Values : tensor([0.0128, 0.0086], grad_fn=<SliceBackward0>) 
    
    Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values : tensor([[-0.0165, -0.0068, -0.0016,  ..., -0.0098,  0.0119,  0.0326],
            [ 0.0330, -0.0306, -0.0129,  ..., -0.0371, -0.0291, -0.0273]],
           grad_fn=<SliceBackward0>) 
    
    Layer: linear_relu_stack.2.bias | Size: torch.Size([512]) | Values : tensor([ 0.0024, -0.0164], grad_fn=<SliceBackward0>) 
    
    Layer: linear_relu_stack.4.weight | Size: torch.Size([10, 512]) | Values : tensor([[ 0.0046,  0.0249,  0.0123,  ...,  0.0352, -0.0170,  0.0232],
            [ 0.0038,  0.0283,  0.0235,  ..., -0.0416,  0.0304,  0.0217]],
           grad_fn=<SliceBackward0>) 
    
    Layer: linear_relu_stack.4.bias | Size: torch.Size([10]) | Values : tensor([0.0118, 0.0417], grad_fn=<SliceBackward0>) 
    


--------------




## Further Reading
- [torch.nn API](https://pytorch.org/docs/stable/nn.html)


