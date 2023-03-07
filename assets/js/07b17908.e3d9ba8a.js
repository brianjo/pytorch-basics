"use strict";(self.webpackChunk=self.webpackChunk||[]).push([[472],{3905:(e,t,n)=>{n.d(t,{Zo:()=>m,kt:()=>h});var o=n(7294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function r(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);t&&(o=o.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,o)}return n}function s(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?r(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):r(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,o,a=function(e,t){if(null==e)return{};var n,o,a={},r=Object.keys(e);for(o=0;o<r.length;o++)n=r[o],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);for(o=0;o<r.length;o++)n=r[o],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var i=o.createContext({}),p=function(e){var t=o.useContext(i),n=t;return e&&(n="function"==typeof e?e(t):s(s({},t),e)),n},m=function(e){var t=p(e.components);return o.createElement(i.Provider,{value:t},e.children)},c="mdxType",d={inlineCode:"code",wrapper:function(e){var t=e.children;return o.createElement(o.Fragment,{},t)}},u=o.forwardRef((function(e,t){var n=e.components,a=e.mdxType,r=e.originalType,i=e.parentName,m=l(e,["components","mdxType","originalType","parentName"]),c=p(n),u=a,h=c["".concat(i,".").concat(u)]||c[u]||d[u]||r;return n?o.createElement(h,s(s({ref:t},m),{},{components:n})):o.createElement(h,s({ref:t},m))}));function h(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var r=n.length,s=new Array(r);s[0]=u;var l={};for(var i in t)hasOwnProperty.call(t,i)&&(l[i]=t[i]);l.originalType=e,l[c]="string"==typeof e?e:a,s[1]=l;for(var p=2;p<r;p++)s[p]=n[p];return o.createElement.apply(null,s)}return o.createElement.apply(null,n)}u.displayName="MDXCreateElement"},9612:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>i,contentTitle:()=>s,default:()=>d,frontMatter:()=>r,metadata:()=>l,toc:()=>p});var o=n(7462),a=(n(7294),n(3905));const r={},s=void 0,l={unversionedId:"Optimization",id:"Optimization",title:"Optimization",description:"Learn the Basics ||",source:"@site/docs/08-Optimization.md",sourceDirName:".",slug:"/Optimization",permalink:"/pytorch-basics/docs/Optimization",draft:!1,tags:[],version:"current",sidebarPosition:8,frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Autograd",permalink:"/pytorch-basics/docs/Autograd"},next:{title:"SaveLoad",permalink:"/pytorch-basics/docs/SaveLoad"}},i={},p=[{value:"Prerequisite Code",id:"prerequisite-code",level:2},{value:"Hyperparameters",id:"hyperparameters",level:2},{value:"Optimization Loop",id:"optimization-loop",level:2},{value:"Loss Function",id:"loss-function",level:3},{value:"Optimizer",id:"optimizer",level:3},{value:"Full Implementation",id:"full-implementation",level:2},{value:"Further Reading",id:"further-reading",level:2}],m={toc:p},c="wrapper";function d(e){let{components:t,...n}=e;return(0,a.kt)(c,(0,o.Z)({},m,n,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("p",null,(0,a.kt)("a",{parentName:"p",href:"intro.html"},"Learn the Basics")," ||\n",(0,a.kt)("a",{parentName:"p",href:"quickstart_tutorial.html"},"Quickstart")," ||\n",(0,a.kt)("a",{parentName:"p",href:"tensorqs_tutorial.html"},"Tensors")," ||\n",(0,a.kt)("a",{parentName:"p",href:"data_tutorial.html"},"Datasets & DataLoaders")," ||\n",(0,a.kt)("a",{parentName:"p",href:"transforms_tutorial.html"},"Transforms")," ||\n",(0,a.kt)("a",{parentName:"p",href:"buildmodel_tutorial.html"},"Build Model")," ||\n",(0,a.kt)("a",{parentName:"p",href:"autogradqs_tutorial.html"},"Autograd")," ||\n",(0,a.kt)("strong",{parentName:"p"},"Optimization")," ||\n",(0,a.kt)("a",{parentName:"p",href:"saveloadrun_tutorial.html"},"Save & Load Model")),(0,a.kt)("h1",{id:"optimizing-model-parameters"},"Optimizing Model Parameters"),(0,a.kt)("p",null,"Now that we have a model and data it's time to train, validate and test our model by optimizing its parameters on\nour data. Training a model is an iterative process; in each iteration the model makes a guess about the output, calculates\nthe error in its guess (",(0,a.kt)("em",{parentName:"p"},"loss"),"), collects the derivatives of the error with respect to its parameters (as we saw in\nthe ",(0,a.kt)("a",{parentName:"p",href:"autograd_tutorial.html"},"previous section"),"), and ",(0,a.kt)("strong",{parentName:"p"},"optimizes")," these parameters using gradient descent. For a more\ndetailed walkthrough of this process, check out this video on ",(0,a.kt)("a",{parentName:"p",href:"https://www.youtube.com/watch?v=tIeHLnjs5U8"},"backpropagation from 3Blue1Brown"),"_."),(0,a.kt)("h2",{id:"prerequisite-code"},"Prerequisite Code"),(0,a.kt)("p",null,"We load the code from the previous sections on ",(0,a.kt)("a",{parentName:"p",href:"data_tutorial.html"},"Datasets & DataLoaders"),"\nand ",(0,a.kt)("a",{parentName:"p",href:"buildmodel_tutorial.html"},"Build Model"),"."),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},'%matplotlib inline\n\nimport torch\nfrom torch import nn\nfrom torch.utils.data import DataLoader\nfrom torchvision import datasets\nfrom torchvision.transforms import ToTensor\n\ntraining_data = datasets.FashionMNIST(\n    root="data",\n    train=True,\n    download=True,\n    transform=ToTensor()\n)\n\ntest_data = datasets.FashionMNIST(\n    root="data",\n    train=False,\n    download=True,\n    transform=ToTensor()\n)\n\ntrain_dataloader = DataLoader(training_data, batch_size=64)\ntest_dataloader = DataLoader(test_data, batch_size=64)\n\nclass NeuralNetwork(nn.Module):\n    def __init__(self):\n        super(NeuralNetwork, self).__init__()\n        self.flatten = nn.Flatten()\n        self.linear_relu_stack = nn.Sequential(\n            nn.Linear(28*28, 512),\n            nn.ReLU(),\n            nn.Linear(512, 512),\n            nn.ReLU(),\n            nn.Linear(512, 10),\n        )\n\n    def forward(self, x):\n        x = self.flatten(x)\n        logits = self.linear_relu_stack(x)\n        return logits\n\nmodel = NeuralNetwork()\n')),(0,a.kt)("h2",{id:"hyperparameters"},"Hyperparameters"),(0,a.kt)("p",null,"Hyperparameters are adjustable parameters that let you control the model optimization process.\nDifferent hyperparameter values can impact model training and convergence rates\n(",(0,a.kt)("a",{parentName:"p",href:"https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html"},"read more"),"_ about hyperparameter tuning)"),(0,a.kt)("p",null,"We define the following hyperparameters for training:"),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"Number of Epochs")," - the number times to iterate over the dataset"),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"Batch Size")," - the number of data samples propagated through the network before the parameters are updated"),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"Learning Rate")," - how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training.")),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},"learning_rate = 1e-3\nbatch_size = 64\nepochs = 5\n")),(0,a.kt)("h2",{id:"optimization-loop"},"Optimization Loop"),(0,a.kt)("p",null,"Once we set our hyperparameters, we can then train and optimize our model with an optimization loop. Each\niteration of the optimization loop is called an ",(0,a.kt)("strong",{parentName:"p"},"epoch"),"."),(0,a.kt)("p",null,"Each epoch consists of two main parts:"),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"The Train Loop")," - iterate over the training dataset and try to converge to optimal parameters."),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"The Validation/Test Loop")," - iterate over the test dataset to check if model performance is improving.")),(0,a.kt)("p",null,"Let's briefly familiarize ourselves with some of the concepts used in the training loop. Jump ahead to\nsee the ",(0,a.kt)("inlineCode",{parentName:"p"},"full-impl-label")," of the optimization loop."),(0,a.kt)("h3",{id:"loss-function"},"Loss Function"),(0,a.kt)("p",null,"When presented with some training data, our untrained network is likely not to give the correct\nanswer. ",(0,a.kt)("strong",{parentName:"p"},"Loss function")," measures the degree of dissimilarity of obtained result to the target value,\nand it is the loss function that we want to minimize during training. To calculate the loss we make a\nprediction using the inputs of our given data sample and compare it against the true data label value."),(0,a.kt)("p",null,"Common loss functions include ",(0,a.kt)("a",{parentName:"p",href:"https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss"},"nn.MSELoss")," (Mean Square Error) for regression tasks, and\n",(0,a.kt)("a",{parentName:"p",href:"https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss"},"nn.NLLLoss")," (Negative Log Likelihood) for classification.\n",(0,a.kt)("a",{parentName:"p",href:"https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss"},"nn.CrossEntropyLoss")," combines ",(0,a.kt)("inlineCode",{parentName:"p"},"nn.LogSoftmax")," and ",(0,a.kt)("inlineCode",{parentName:"p"},"nn.NLLLoss"),"."),(0,a.kt)("p",null,"We pass our model's output logits to ",(0,a.kt)("inlineCode",{parentName:"p"},"nn.CrossEntropyLoss"),", which will normalize the logits and compute the prediction error."),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},"# Initialize the loss function\nloss_fn = nn.CrossEntropyLoss()\n")),(0,a.kt)("h3",{id:"optimizer"},"Optimizer"),(0,a.kt)("p",null,"Optimization is the process of adjusting model parameters to reduce model error in each training step. ",(0,a.kt)("strong",{parentName:"p"},"Optimization algorithms")," define how this process is performed (in this example we use Stochastic Gradient Descent).\nAll optimization logic is encapsulated in  the ",(0,a.kt)("inlineCode",{parentName:"p"},"optimizer")," object. Here, we use the SGD optimizer; additionally, there are many ",(0,a.kt)("a",{parentName:"p",href:"https://pytorch.org/docs/stable/optim.html"},"different optimizers"),"\navailable in PyTorch such as ADAM and RMSProp, that work better for different kinds of models and data."),(0,a.kt)("p",null,"We initialize the optimizer by registering the model's parameters that need to be trained, and passing in the learning rate hyperparameter."),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},"optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)\n")),(0,a.kt)("p",null,"Inside the training loop, optimization happens in three steps:"),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},"Call ",(0,a.kt)("inlineCode",{parentName:"li"},"optimizer.zero_grad()")," to reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration."),(0,a.kt)("li",{parentName:"ul"},"Backpropagate the prediction loss with a call to ",(0,a.kt)("inlineCode",{parentName:"li"},"loss.backward()"),". PyTorch deposits the gradients of the loss w.r.t. each parameter."),(0,a.kt)("li",{parentName:"ul"},"Once we have our gradients, we call ",(0,a.kt)("inlineCode",{parentName:"li"},"optimizer.step()")," to adjust the parameters by the gradients collected in the backward pass.")),(0,a.kt)("h2",{id:"full-implementation"},"Full Implementation"),(0,a.kt)("p",null,"We define ",(0,a.kt)("inlineCode",{parentName:"p"},"train_loop")," that loops over our optimization code, and ",(0,a.kt)("inlineCode",{parentName:"p"},"test_loop")," that\nevaluates the model's performance against our test data."),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},'def train_loop(dataloader, model, loss_fn, optimizer):\n    size = len(dataloader.dataset)\n    for batch, (X, y) in enumerate(dataloader):\n        # Compute prediction and loss\n        pred = model(X)\n        loss = loss_fn(pred, y)\n\n        # Backpropagation\n        optimizer.zero_grad()\n        loss.backward()\n        optimizer.step()\n\n        if batch % 100 == 0:\n            loss, current = loss.item(), (batch + 1) * len(X)\n            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")\n\n\ndef test_loop(dataloader, model, loss_fn):\n    size = len(dataloader.dataset)\n    num_batches = len(dataloader)\n    test_loss, correct = 0, 0\n\n    with torch.no_grad():\n        for X, y in dataloader:\n            pred = model(X)\n            test_loss += loss_fn(pred, y).item()\n            correct += (pred.argmax(1) == y).type(torch.float).sum().item()\n\n    test_loss /= num_batches\n    correct /= size\n    print(f"Test Error: \\n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \\n")\n')),(0,a.kt)("p",null,"We initialize the loss function and optimizer, and pass it to ",(0,a.kt)("inlineCode",{parentName:"p"},"train_loop")," and ",(0,a.kt)("inlineCode",{parentName:"p"},"test_loop"),".\nFeel free to increase the number of epochs to track the model's improving performance."),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},'loss_fn = nn.CrossEntropyLoss()\noptimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)\n\nepochs = 10\nfor t in range(epochs):\n    print(f"Epoch {t+1}\\n-------------------------------")\n    train_loop(train_dataloader, model, loss_fn, optimizer)\n    test_loop(test_dataloader, model, loss_fn)\nprint("Done!")\n')),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre"},"Epoch 1\n-------------------------------\nloss: 2.300194  [   64/60000]\nloss: 2.288841  [ 6464/60000]\nloss: 2.278213  [12864/60000]\nloss: 2.266712  [19264/60000]\nloss: 2.251664  [25664/60000]\nloss: 2.222698  [32064/60000]\nloss: 2.225873  [38464/60000]\nloss: 2.197677  [44864/60000]\nloss: 2.182950  [51264/60000]\nloss: 2.150051  [57664/60000]\nTest Error: \n Accuracy: 49.7%, Avg loss: 2.145039 \n\nEpoch 2\n-------------------------------\nloss: 2.157795  [   64/60000]\nloss: 2.138097  [ 6464/60000]\nloss: 2.086915  [12864/60000]\nloss: 2.094276  [19264/60000]\nloss: 2.046509  [25664/60000]\nloss: 1.990505  [32064/60000]\nloss: 2.009515  [38464/60000]\nloss: 1.936049  [44864/60000]\nloss: 1.930621  [51264/60000]\nloss: 1.851469  [57664/60000]\nTest Error: \n Accuracy: 53.8%, Avg loss: 1.848206 \n\nEpoch 3\n-------------------------------\nloss: 1.892401  [   64/60000]\nloss: 1.842193  [ 6464/60000]\nloss: 1.735193  [12864/60000]\nloss: 1.767924  [19264/60000]\nloss: 1.664628  [25664/60000]\nloss: 1.626719  [32064/60000]\nloss: 1.643489  [38464/60000]\nloss: 1.556118  [44864/60000]\nloss: 1.578335  [51264/60000]\nloss: 1.465860  [57664/60000]\nTest Error: \n Accuracy: 60.2%, Avg loss: 1.484835 \n\nEpoch 4\n-------------------------------\nloss: 1.560882  [   64/60000]\nloss: 1.511349  [ 6464/60000]\nloss: 1.376379  [12864/60000]\nloss: 1.442743  [19264/60000]\nloss: 1.334723  [25664/60000]\nloss: 1.332103  [32064/60000]\nloss: 1.343892  [38464/60000]\nloss: 1.280821  [44864/60000]\nloss: 1.313386  [51264/60000]\nloss: 1.206355  [57664/60000]\nTest Error: \n Accuracy: 63.7%, Avg loss: 1.234206 \n\nEpoch 5\n-------------------------------\nloss: 1.315912  [   64/60000]\nloss: 1.286868  [ 6464/60000]\nloss: 1.134339  [12864/60000]\nloss: 1.233307  [19264/60000]\nloss: 1.118453  [25664/60000]\nloss: 1.140306  [32064/60000]\nloss: 1.159836  [38464/60000]\nloss: 1.110095  [44864/60000]\nloss: 1.145639  [51264/60000]\nloss: 1.054201  [57664/60000]\nTest Error: \n Accuracy: 65.3%, Avg loss: 1.076539 \n\nEpoch 6\n-------------------------------\nloss: 1.151193  [   64/60000]\nloss: 1.144089  [ 6464/60000]\nloss: 0.972367  [12864/60000]\nloss: 1.099682  [19264/60000]\nloss: 0.982088  [25664/60000]\nloss: 1.010691  [32064/60000]\nloss: 1.045632  [38464/60000]\nloss: 1.000589  [44864/60000]\nloss: 1.034699  [51264/60000]\nloss: 0.958953  [57664/60000]\nTest Error: \n Accuracy: 66.2%, Avg loss: 0.974052 \n\nEpoch 7\n-------------------------------\nloss: 1.036133  [   64/60000]\nloss: 1.051354  [ 6464/60000]\nloss: 0.860921  [12864/60000]\nloss: 1.010238  [19264/60000]\nloss: 0.895602  [25664/60000]\nloss: 0.919513  [32064/60000]\nloss: 0.971160  [38464/60000]\nloss: 0.929603  [44864/60000]\nloss: 0.957854  [51264/60000]\nloss: 0.895215  [57664/60000]\nTest Error: \n Accuracy: 67.3%, Avg loss: 0.904191 \n\nEpoch 8\n-------------------------------\nloss: 0.951518  [   64/60000]\nloss: 0.986645  [ 6464/60000]\nloss: 0.780995  [12864/60000]\nloss: 0.946828  [19264/60000]\nloss: 0.838171  [25664/60000]\nloss: 0.852497  [32064/60000]\nloss: 0.918538  [38464/60000]\nloss: 0.881657  [44864/60000]\nloss: 0.902239  [51264/60000]\nloss: 0.849078  [57664/60000]\nTest Error: \n Accuracy: 68.5%, Avg loss: 0.853808 \n\nEpoch 9\n-------------------------------\nloss: 0.886171  [   64/60000]\nloss: 0.937722  [ 6464/60000]\nloss: 0.721067  [12864/60000]\nloss: 0.899385  [19264/60000]\nloss: 0.797614  [25664/60000]\nloss: 0.801852  [32064/60000]\nloss: 0.878401  [38464/60000]\nloss: 0.847703  [44864/60000]\nloss: 0.860428  [51264/60000]\nloss: 0.813565  [57664/60000]\nTest Error: \n Accuracy: 69.6%, Avg loss: 0.815620 \n\nEpoch 10\n-------------------------------\nloss: 0.833724  [   64/60000]\nloss: 0.898364  [ 6464/60000]\nloss: 0.674373  [12864/60000]\nloss: 0.862498  [19264/60000]\nloss: 0.767404  [25664/60000]\nloss: 0.762986  [32064/60000]\nloss: 0.845986  [38464/60000]\nloss: 0.822418  [44864/60000]\nloss: 0.828057  [51264/60000]\nloss: 0.784834  [57664/60000]\nTest Error: \n Accuracy: 70.6%, Avg loss: 0.785303 \n\nDone!\n")),(0,a.kt)("h2",{id:"further-reading"},"Further Reading"),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"https://pytorch.org/docs/stable/nn.html#loss-functions"},"Loss Functions")),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"https://pytorch.org/docs/stable/optim.html"},"torch.optim")),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"https://pytorch.org/tutorials/recipes/recipes/warmstarting_model_using_parameters_from_a_different_model.html"},"Warmstart Training a Model"))))}d.isMDXComponent=!0}}]);