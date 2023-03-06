"use strict";(self.webpackChunk=self.webpackChunk||[]).push([[576],{3905:(e,t,a)=>{a.d(t,{Zo:()=>d,kt:()=>g});var n=a(7294);function A(e,t,a){return t in e?Object.defineProperty(e,t,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[t]=a,e}function r(e,t){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),a.push.apply(a,n)}return a}function i(e){for(var t=1;t<arguments.length;t++){var a=null!=arguments[t]?arguments[t]:{};t%2?r(Object(a),!0).forEach((function(t){A(e,t,a[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):r(Object(a)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(a,t))}))}return e}function o(e,t){if(null==e)return{};var a,n,A=function(e,t){if(null==e)return{};var a,n,A={},r=Object.keys(e);for(n=0;n<r.length;n++)a=r[n],t.indexOf(a)>=0||(A[a]=e[a]);return A}(e,t);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);for(n=0;n<r.length;n++)a=r[n],t.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(A[a]=e[a])}return A}var s=n.createContext({}),l=function(e){var t=n.useContext(s),a=t;return e&&(a="function"==typeof e?e(t):i(i({},t),e)),a},d=function(e){var t=l(e.components);return n.createElement(s.Provider,{value:t},e.children)},p="mdxType",m={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},u=n.forwardRef((function(e,t){var a=e.components,A=e.mdxType,r=e.originalType,s=e.parentName,d=o(e,["components","mdxType","originalType","parentName"]),p=l(a),u=A,g=p["".concat(s,".").concat(u)]||p[u]||m[u]||r;return a?n.createElement(g,i(i({ref:t},d),{},{components:a})):n.createElement(g,i({ref:t},d))}));function g(e,t){var a=arguments,A=t&&t.mdxType;if("string"==typeof e||A){var r=a.length,i=new Array(r);i[0]=u;var o={};for(var s in t)hasOwnProperty.call(t,s)&&(o[s]=t[s]);o.originalType=e,o[p]="string"==typeof e?e:A,i[1]=o;for(var l=2;l<r;l++)i[l]=a[l];return n.createElement.apply(null,i)}return n.createElement.apply(null,a)}u.displayName="MDXCreateElement"},5138:(e,t,a)=>{a.r(t),a.d(t,{assets:()=>s,contentTitle:()=>i,default:()=>m,frontMatter:()=>r,metadata:()=>o,toc:()=>l});var n=a(7462),A=(a(7294),a(3905));const r={},i=void 0,o={unversionedId:"Data",id:"Data",title:"Data",description:"Learn the Basics ||",source:"@site/docs/04-Data.md",sourceDirName:".",slug:"/Data",permalink:"/pytorch-basics/docs/Data",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/04-Data.md",tags:[],version:"current",sidebarPosition:4,frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Tensors",permalink:"/pytorch-basics/docs/Tensors"},next:{title:"Transforms",permalink:"/pytorch-basics/docs/Transforms"}},s={},l=[{value:"Loading a Dataset",id:"loading-a-dataset",level:2},{value:"Iterating and Visualizing the Dataset",id:"iterating-and-visualizing-the-dataset",level:2},{value:"Creating a Custom Dataset for your files",id:"creating-a-custom-dataset-for-your-files",level:2},{value:"<strong>init</strong>",id:"init",level:3},{value:"<strong>len</strong>",id:"len",level:3},{value:"<strong>getitem</strong>",id:"getitem",level:3},{value:"Preparing your data for training with DataLoaders",id:"preparing-your-data-for-training-with-dataloaders",level:2},{value:"Iterate through the DataLoader",id:"iterate-through-the-dataloader",level:2},{value:"Further Reading",id:"further-reading",level:2}],d={toc:l},p="wrapper";function m(e){let{components:t,...r}=e;return(0,A.kt)(p,(0,n.Z)({},d,r,{components:t,mdxType:"MDXLayout"}),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre",className:"language-python"},"%matplotlib inline\n")),(0,A.kt)("p",null,(0,A.kt)("a",{parentName:"p",href:"intro.html"},"Learn the Basics")," ||\n",(0,A.kt)("a",{parentName:"p",href:"quickstart_tutorial.html"},"Quickstart")," ||\n",(0,A.kt)("a",{parentName:"p",href:"tensorqs_tutorial.html"},"Tensors")," ||\n",(0,A.kt)("strong",{parentName:"p"},"Datasets & DataLoaders")," ||\n",(0,A.kt)("a",{parentName:"p",href:"transforms_tutorial.html"},"Transforms")," ||\n",(0,A.kt)("a",{parentName:"p",href:"buildmodel_tutorial.html"},"Build Model")," ||\n",(0,A.kt)("a",{parentName:"p",href:"autogradqs_tutorial.html"},"Autograd")," ||\n",(0,A.kt)("a",{parentName:"p",href:"optimization_tutorial.html"},"Optimization")," ||\n",(0,A.kt)("a",{parentName:"p",href:"saveloadrun_tutorial.html"},"Save & Load Model")),(0,A.kt)("h1",{id:"datasets--dataloaders"},"Datasets & DataLoaders"),(0,A.kt)("p",null,"Code for processing data samples can get messy and hard to maintain; we ideally want our dataset code\nto be decoupled from our model training code for better readability and modularity.\nPyTorch provides two data primitives: ",(0,A.kt)("inlineCode",{parentName:"p"},"torch.utils.data.DataLoader")," and ",(0,A.kt)("inlineCode",{parentName:"p"},"torch.utils.data.Dataset"),"\nthat allow you to use pre-loaded datasets as well as your own data.\n",(0,A.kt)("inlineCode",{parentName:"p"},"Dataset")," stores the samples and their corresponding labels, and ",(0,A.kt)("inlineCode",{parentName:"p"},"DataLoader")," wraps an iterable around\nthe ",(0,A.kt)("inlineCode",{parentName:"p"},"Dataset")," to enable easy access to the samples."),(0,A.kt)("p",null,"PyTorch domain libraries provide a number of pre-loaded datasets (such as FashionMNIST) that\nsubclass ",(0,A.kt)("inlineCode",{parentName:"p"},"torch.utils.data.Dataset")," and implement functions specific to the particular data.\nThey can be used to prototype and benchmark your model. You can find them\nhere: ",(0,A.kt)("a",{parentName:"p",href:"https://pytorch.org/vision/stable/datasets.html"},"Image Datasets"),",\n",(0,A.kt)("a",{parentName:"p",href:"https://pytorch.org/text/stable/datasets.html"},"Text Datasets"),", and\n",(0,A.kt)("a",{parentName:"p",href:"https://pytorch.org/audio/stable/datasets.html"},"Audio Datasets")),(0,A.kt)("h2",{id:"loading-a-dataset"},"Loading a Dataset"),(0,A.kt)("p",null,"Here is an example of how to load the ",(0,A.kt)("a",{parentName:"p",href:"https://research.zalando.com/project/fashion_mnist/fashion_mnist/"},"Fashion-MNIST")," dataset from TorchVision.\nFashion-MNIST is a dataset of Zalando\u2019s article images consisting of 60,000 training examples and 10,000 test examples.\nEach example comprises a 28\xd728 grayscale image and an associated label from one of 10 classes."),(0,A.kt)("p",null,"We load the ",(0,A.kt)("a",{parentName:"p",href:"https://pytorch.org/vision/stable/datasets.html#fashion-mnist"},"FashionMNIST Dataset")," with the following parameters:"),(0,A.kt)("ul",null,(0,A.kt)("li",{parentName:"ul"},(0,A.kt)("inlineCode",{parentName:"li"},"root")," is the path where the train/test data is stored,"),(0,A.kt)("li",{parentName:"ul"},(0,A.kt)("inlineCode",{parentName:"li"},"train")," specifies training or test dataset,"),(0,A.kt)("li",{parentName:"ul"},(0,A.kt)("inlineCode",{parentName:"li"},"download=True")," downloads the data from the internet if it's not available at ",(0,A.kt)("inlineCode",{parentName:"li"},"root"),"."),(0,A.kt)("li",{parentName:"ul"},(0,A.kt)("inlineCode",{parentName:"li"},"transform")," and ",(0,A.kt)("inlineCode",{parentName:"li"},"target_transform")," specify the feature and label transformations")),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre",className:"language-python"},'import torch\nfrom torch.utils.data import Dataset\nfrom torchvision import datasets\nfrom torchvision.transforms import ToTensor\nimport matplotlib.pyplot as plt\n\n\ntraining_data = datasets.FashionMNIST(\n    root="data",\n    train=True,\n    download=True,\n    transform=ToTensor()\n)\n\ntest_data = datasets.FashionMNIST(\n    root="data",\n    train=False,\n    download=True,\n    transform=ToTensor()\n)\n')),(0,A.kt)("h2",{id:"iterating-and-visualizing-the-dataset"},"Iterating and Visualizing the Dataset"),(0,A.kt)("p",null,"We can index ",(0,A.kt)("inlineCode",{parentName:"p"},"Datasets")," manually like a list: ",(0,A.kt)("inlineCode",{parentName:"p"},"training_data[index]"),".\nWe use ",(0,A.kt)("inlineCode",{parentName:"p"},"matplotlib")," to visualize some samples in our training data."),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre",className:"language-python"},'labels_map = {\n    0: "T-Shirt",\n    1: "Trouser",\n    2: "Pullover",\n    3: "Dress",\n    4: "Coat",\n    5: "Sandal",\n    6: "Shirt",\n    7: "Sneaker",\n    8: "Bag",\n    9: "Ankle Boot",\n}\nfigure = plt.figure(figsize=(8, 8))\ncols, rows = 3, 3\nfor i in range(1, cols * rows + 1):\n    sample_idx = torch.randint(len(training_data), size=(1,)).item()\n    img, label = training_data[sample_idx]\n    figure.add_subplot(rows, cols, i)\n    plt.title(labels_map[label])\n    plt.axis("off")\n    plt.imshow(img.squeeze(), cmap="gray")\nplt.show()\n')),(0,A.kt)("p",null,(0,A.kt)("img",{alt:"png",src:a(5558).Z,width:"638",height:"658"})),(0,A.kt)("p",null,"..\n.. figure:: /_static/img/basics/fashion_mnist.png\n:alt: fashion_mnist"),(0,A.kt)("hr",null),(0,A.kt)("h2",{id:"creating-a-custom-dataset-for-your-files"},"Creating a Custom Dataset for your files"),(0,A.kt)("p",null,"A custom Dataset class must implement three functions: ",(0,A.kt)("inlineCode",{parentName:"p"},"__init__"),", ",(0,A.kt)("inlineCode",{parentName:"p"},"__len__"),", and ",(0,A.kt)("inlineCode",{parentName:"p"},"__getitem__"),".\nTake a look at this implementation; the FashionMNIST images are stored\nin a directory ",(0,A.kt)("inlineCode",{parentName:"p"},"img_dir"),", and their labels are stored separately in a CSV file ",(0,A.kt)("inlineCode",{parentName:"p"},"annotations_file"),"."),(0,A.kt)("p",null,"In the next sections, we'll break down what's happening in each of these functions."),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre",className:"language-python"},"import os\nimport pandas as pd\nfrom torchvision.io import read_image\n\nclass CustomImageDataset(Dataset):\n    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):\n        self.img_labels = pd.read_csv(annotations_file)\n        self.img_dir = img_dir\n        self.transform = transform\n        self.target_transform = target_transform\n\n    def __len__(self):\n        return len(self.img_labels)\n\n    def __getitem__(self, idx):\n        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])\n        image = read_image(img_path)\n        label = self.img_labels.iloc[idx, 1]\n        if self.transform:\n            image = self.transform(image)\n        if self.target_transform:\n            label = self.target_transform(label)\n        return image, label\n")),(0,A.kt)("h3",{id:"init"},(0,A.kt)("strong",{parentName:"h3"},"init")),(0,A.kt)("p",null,"The ",(0,A.kt)("strong",{parentName:"p"},"init")," function is run once when instantiating the Dataset object. We initialize\nthe directory containing the images, the annotations file, and both transforms (covered\nin more detail in the next section)."),(0,A.kt)("p",null,"The labels.csv file looks like: ::"),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre"},"tshirt1.jpg, 0\ntshirt2.jpg, 0\n......\nankleboot999.jpg, 9\n")),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre",className:"language-python"},"def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):\n    self.img_labels = pd.read_csv(annotations_file)\n    self.img_dir = img_dir\n    self.transform = transform\n    self.target_transform = target_transform\n")),(0,A.kt)("h3",{id:"len"},(0,A.kt)("strong",{parentName:"h3"},"len")),(0,A.kt)("p",null,"The ",(0,A.kt)("strong",{parentName:"p"},"len")," function returns the number of samples in our dataset."),(0,A.kt)("p",null,"Example:"),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre",className:"language-python"},"def __len__(self):\n    return len(self.img_labels)\n")),(0,A.kt)("h3",{id:"getitem"},(0,A.kt)("strong",{parentName:"h3"},"getitem")),(0,A.kt)("p",null,"The ",(0,A.kt)("strong",{parentName:"p"},"getitem")," function loads and returns a sample from the dataset at the given index ",(0,A.kt)("inlineCode",{parentName:"p"},"idx"),".\nBased on the index, it identifies the image's location on disk, converts that to a tensor using ",(0,A.kt)("inlineCode",{parentName:"p"},"read_image"),", retrieves the\ncorresponding label from the csv data in ",(0,A.kt)("inlineCode",{parentName:"p"},"self.img_labels"),", calls the transform functions on them (if applicable), and returns the\ntensor image and corresponding label in a tuple."),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre",className:"language-python"},"def __getitem__(self, idx):\n    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])\n    image = read_image(img_path)\n    label = self.img_labels.iloc[idx, 1]\n    if self.transform:\n        image = self.transform(image)\n    if self.target_transform:\n        label = self.target_transform(label)\n    return image, label\n")),(0,A.kt)("hr",null),(0,A.kt)("h2",{id:"preparing-your-data-for-training-with-dataloaders"},"Preparing your data for training with DataLoaders"),(0,A.kt)("p",null,"The ",(0,A.kt)("inlineCode",{parentName:"p"},"Dataset")," retrieves our dataset's features and labels one sample at a time. While training a model, we typically want to\npass samples in \"minibatches\", reshuffle the data at every epoch to reduce model overfitting, and use Python's ",(0,A.kt)("inlineCode",{parentName:"p"},"multiprocessing")," to\nspeed up data retrieval."),(0,A.kt)("p",null,(0,A.kt)("inlineCode",{parentName:"p"},"DataLoader")," is an iterable that abstracts this complexity for us in an easy API."),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre",className:"language-python"},"from torch.utils.data import DataLoader\n\ntrain_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)\ntest_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)\n")),(0,A.kt)("h2",{id:"iterate-through-the-dataloader"},"Iterate through the DataLoader"),(0,A.kt)("p",null,"We have loaded that dataset into the ",(0,A.kt)("inlineCode",{parentName:"p"},"DataLoader")," and can iterate through the dataset as needed.\nEach iteration below returns a batch of ",(0,A.kt)("inlineCode",{parentName:"p"},"train_features")," and ",(0,A.kt)("inlineCode",{parentName:"p"},"train_labels")," (containing ",(0,A.kt)("inlineCode",{parentName:"p"},"batch_size=64")," features and labels respectively).\nBecause we specified ",(0,A.kt)("inlineCode",{parentName:"p"},"shuffle=True"),", after we iterate over all batches the data is shuffled (for finer-grained control over\nthe data loading order, take a look at ",(0,A.kt)("a",{parentName:"p",href:"https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler"},"Samplers"),")."),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre",className:"language-python"},'# Display image and label.\ntrain_features, train_labels = next(iter(train_dataloader))\nprint(f"Feature batch shape: {train_features.size()}")\nprint(f"Labels batch shape: {train_labels.size()}")\nimg = train_features[0].squeeze()\nlabel = train_labels[0]\nplt.imshow(img, cmap="gray")\nplt.show()\nprint(f"Label: {label}")\n')),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre"},"Feature batch shape: torch.Size([64, 1, 28, 28])\nLabels batch shape: torch.Size([64])\n")),(0,A.kt)("p",null,(0,A.kt)("img",{alt:"png",src:a(2920).Z,width:"416",height:"413"})),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre"},"Label: 6\n")),(0,A.kt)("hr",null),(0,A.kt)("h2",{id:"further-reading"},"Further Reading"),(0,A.kt)("ul",null,(0,A.kt)("li",{parentName:"ul"},(0,A.kt)("a",{parentName:"li",href:"https://pytorch.org/docs/stable/data.html"},"torch.utils.data API"))))}m.isMDXComponent=!0},2920:(e,t,a)=>{a.d(t,{Z:()=>n});const n="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAaAAAAGdCAYAAABU0qcqAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAeT0lEQVR4nO3df2xV9f3H8ddtaW8LlFva2l+jQAsqKj8WmXQNylAaSs2MKFnwRzIwBqIrZsj8kS4qsi2pw0TNXAdbsoEm4q9MIJqFBaqUuAELKCHEraFYRwm0aA0tlP6i93z/IHbfKwX8HO6979vyfCQnae897543n57y6uk9fTfgeZ4nAADiLMm6AQDA1YkAAgCYIIAAACYIIACACQIIAGCCAAIAmCCAAAAmCCAAgIkR1g18Wzgc1vHjx5WRkaFAIGDdDgDAked5On36tAoLC5WUdPHrnIQLoOPHj6uoqMi6DQDAFWpubta4ceMu+nzCBVBGRoZ1C1cdv1eaiTzF6Y477vBVV1BQ4FwzatQo55q0tDTnmsLCQueaP/3pT841kvT555/7qgP+v8v9fx6zAKqtrdWLL76olpYWzZgxQ6+++qpmzZp12Tp+7HZl/KzfcAygESP8ndqpqalxqQkGg841fkLrUj/+SATx+npP5HN1OLvc5zcmZ+fbb7+tVatWafXq1frkk080Y8YMVVRU6OTJk7E4HABgCIpJAL300ktatmyZHnroId14441av369Ro4cqb/85S+xOBwAYAiKegD19vZq//79Ki8v/99BkpJUXl6u3bt3X7B/T0+POjo6IjYAwPAX9QD66quv1N/fr7y8vIjH8/Ly1NLScsH+NTU1CoVCAxt3wAHA1cH8Fcrq6mq1t7cPbM3NzdYtAQDiIOp3weXk5Cg5OVmtra0Rj7e2tio/P/+C/YPBoK87ggAAQ1vUr4BSU1M1c+ZM1dXVDTwWDodVV1ensrKyaB8OADBExeT3gFatWqUlS5boBz/4gWbNmqVXXnlFnZ2deuihh2JxOADAEBSTAFq8eLG+/PJLPffcc2ppadH3v/99bdu27YIbEwAAV6+Al2C/ItzR0aFQKGTdxpDl5zfLE+wUuMAf//hH55rly5f7OlZ/f79zzcaNG51rbrzxRucaP6+VZmZmOtdI0osvvuhcs379el/HiofhOO1jKGhvb9eYMWMu+rz5XXAAgKsTAQQAMEEAAQBMEEAAABMEEADABAEEADBBAAEATBBAAAATBBAAwAQBBAAwQQABAEwQQAAAEwwjhX784x/7qvPz5zXGjRvnXJOdne1c88UXXzjXSFJbW5tzjZ8hod3d3c41fgalDvZHIL+L3Nxc55qTJ0861zQ2NjrXrFmzxrnm+PHjzjW4cgwjBQAkJAIIAGCCAAIAmCCAAAAmCCAAgAkCCABgggACAJgggAAAJgggAIAJAggAYIIAAgCYIIAAACYIIACACaZhQx999JGvOj+Tlr/66ivnmqQk9++T0tLSnGskKTk52bnmySefdK658847nWvKy8uda86cOeNcI53/OnSVnp7uXDNx4kTnmnfeece55oknnnCuwZVjGjYAICERQAAAEwQQAMAEAQQAMEEAAQBMEEAAABMEEADABAEEADBBAAEATBBAAAATBBAAwAQBBAAwMcK6Adj78ssvfdWNGjXKuebcuXPONf39/c41gUDAuUaSiouLnWu2b9/uXJOTk+Nc89Of/tS55rPPPnOukfx9nrq6upxrWlpanGt27drlXIPExBUQAMAEAQQAMEEAAQBMEEAAABMEEADABAEEADBBAAEATBBAAAATBBAAwAQBBAAwQQABAEwQQAAAEwwjhU6fPu2rLjk52bkmHA77OpYrP8M0r6TOVWpqqnNNT0+Pc01nZ6dzjSSlpKQ41/j53PoZNNvb2+tcg8TEFRAAwAQBBAAwEfUAev755xUIBCK2KVOmRPswAIAhLiavAd10003asWPH/w4ygpeaAACRYpIMI0aMUH5+fiw+NABgmIjJa0CHDx9WYWGhSkpK9OCDD+ro0aMX3benp0cdHR0RGwBg+It6AJWWlmrjxo3atm2b1q1bp6amJt12220XvdW3pqZGoVBoYCsqKop2SwCABBT1AKqsrNRPfvITTZ8+XRUVFfrb3/6mU6dO6Z133hl0/+rqarW3tw9szc3N0W4JAJCAYn53QGZmpq677jo1NjYO+nwwGFQwGIx1GwCABBPz3wM6c+aMjhw5ooKCglgfCgAwhEQ9gJ544gnV19friy++0D//+U/dc889Sk5O1v333x/tQwEAhrCo/wju2LFjuv/++9XW1qZrrrlGt956q/bs2aNrrrkm2ocCAAxhUQ+gt956K9ofEjHmd2BlUpL7BbSfYZ9+fpHZ74BVP0NC/cjK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0AcffKCPPvoo4s+35Ofnq7e3V6dOnYrYf7ieDxdbh8GUlpZKUkKdDwkfQKmpqZo5c6bq6uoGHguHw6qrq1NZWZlhZ/bOnDmjI0eOqKCgwLoVM8XFxcrPz484Pzo6OrR3796r/vw4duyY2trahtX54XmeVqxYoc2bN+vDDz9UcXFxxPMzZ85USkpKxPnQ0NCgo0ePDqvz4XLrMJgDBw5IUmKdD9Z3QXwXb731lhcMBr2NGzd6n332mbd8+XIvMzPTa2lpsW4trn7xi194O3fu9Jqamrx//OMfXnl5uZeTk+OdPHnSurWYOn36tPfpp596n376qSfJe+mll7xPP/3U++9//+t5nue98MILXmZmprd161bv4MGD3t133+0VFxd7XV1dxp1H16XW4fTp094TTzzh7d6922tqavJ27Njh3Xzzzd61117rdXd3W7ceNY8++qgXCoW8nTt3eidOnBjYzp49O7DPI4884o0fP9778MMPvX379nllZWVeWVmZYdfRd7l1aGxs9H71q195+/bt85qamrytW7d6JSUl3pw5c4w7jzQkAsjzPO/VV1/1xo8f76WmpnqzZs3y9uzZY91S3C1evNgrKCjwUlNTve9973ve4sWLvcbGRuu2Yu6jjz7yJF2wLVmyxPO887diP/vss15eXp4XDAa9efPmeQ0NDbZNx8Cl1uHs2bPe/PnzvWuuucZLSUnxJkyY4C1btmzYfZM22L9fkrdhw4aBfbq6uryf/exn3tixY72RI0d699xzj3fixAm7pmPgcutw9OhRb86cOV5WVpYXDAa9yZMne08++aTX3t5u2/i38OcYAAAmEv41IADA8EQAAQBMEEAAABMEEADABAEEADBBAAEATBBAAAATBBAAwAQBBAAwQQABAEwQQAAAEwQQAMDE/wE4VN8Do91ANgAAAABJRU5ErkJggg=="},5558:(e,t,a)=>{a.d(t,{Z:()=>n});const n=a.p+"assets/images/04-Data_6_0-e8f5b3d0fb7d7daeb7926873fd6564b8.png"}}]);