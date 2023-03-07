"use strict";(self.webpackChunk=self.webpackChunk||[]).push([[576],{3905:(e,t,a)=>{a.d(t,{Zo:()=>d,kt:()=>u});var n=a(7294);function A(e,t,a){return t in e?Object.defineProperty(e,t,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[t]=a,e}function r(e,t){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),a.push.apply(a,n)}return a}function i(e){for(var t=1;t<arguments.length;t++){var a=null!=arguments[t]?arguments[t]:{};t%2?r(Object(a),!0).forEach((function(t){A(e,t,a[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):r(Object(a)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(a,t))}))}return e}function l(e,t){if(null==e)return{};var a,n,A=function(e,t){if(null==e)return{};var a,n,A={},r=Object.keys(e);for(n=0;n<r.length;n++)a=r[n],t.indexOf(a)>=0||(A[a]=e[a]);return A}(e,t);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);for(n=0;n<r.length;n++)a=r[n],t.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(A[a]=e[a])}return A}var o=n.createContext({}),s=function(e){var t=n.useContext(o),a=t;return e&&(a="function"==typeof e?e(t):i(i({},t),e)),a},d=function(e){var t=s(e.components);return n.createElement(o.Provider,{value:t},e.children)},p="mdxType",m={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},g=n.forwardRef((function(e,t){var a=e.components,A=e.mdxType,r=e.originalType,o=e.parentName,d=l(e,["components","mdxType","originalType","parentName"]),p=s(a),g=A,u=p["".concat(o,".").concat(g)]||p[g]||m[g]||r;return a?n.createElement(u,i(i({ref:t},d),{},{components:a})):n.createElement(u,i({ref:t},d))}));function u(e,t){var a=arguments,A=t&&t.mdxType;if("string"==typeof e||A){var r=a.length,i=new Array(r);i[0]=g;var l={};for(var o in t)hasOwnProperty.call(t,o)&&(l[o]=t[o]);l.originalType=e,l[p]="string"==typeof e?e:A,i[1]=l;for(var s=2;s<r;s++)i[s]=a[s];return n.createElement.apply(null,i)}return n.createElement.apply(null,a)}g.displayName="MDXCreateElement"},5138:(e,t,a)=>{a.r(t),a.d(t,{assets:()=>o,contentTitle:()=>i,default:()=>m,frontMatter:()=>r,metadata:()=>l,toc:()=>s});var n=a(7462),A=(a(7294),a(3905));const r={},i=void 0,l={unversionedId:"Data",id:"Data",title:"Data",description:"Learn the Basics ||",source:"@site/docs/04-Data.md",sourceDirName:".",slug:"/Data",permalink:"/pytorch-basics/docs/Data",draft:!1,tags:[],version:"current",sidebarPosition:4,frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Tensors",permalink:"/pytorch-basics/docs/Tensors"},next:{title:"Transforms",permalink:"/pytorch-basics/docs/Transforms"}},o={},s=[{value:"Loading a Dataset",id:"loading-a-dataset",level:2},{value:"Iterating and Visualizing the Dataset",id:"iterating-and-visualizing-the-dataset",level:2},{value:"Creating a Custom Dataset for your files",id:"creating-a-custom-dataset-for-your-files",level:2},{value:"<strong>init</strong>",id:"init",level:3},{value:"<strong>len</strong>",id:"len",level:3},{value:"<strong>getitem</strong>",id:"getitem",level:3},{value:"Preparing your data for training with DataLoaders",id:"preparing-your-data-for-training-with-dataloaders",level:2},{value:"Iterate through the DataLoader",id:"iterate-through-the-dataloader",level:2},{value:"Further Reading",id:"further-reading",level:2}],d={toc:s},p="wrapper";function m(e){let{components:t,...r}=e;return(0,A.kt)(p,(0,n.Z)({},d,r,{components:t,mdxType:"MDXLayout"}),(0,A.kt)("p",null,(0,A.kt)("a",{parentName:"p",href:"intro.html"},"Learn the Basics")," ||\n",(0,A.kt)("a",{parentName:"p",href:"quickstart_tutorial.html"},"Quickstart")," ||\n",(0,A.kt)("a",{parentName:"p",href:"tensorqs_tutorial.html"},"Tensors")," ||\n",(0,A.kt)("strong",{parentName:"p"},"Datasets & DataLoaders")," ||\n",(0,A.kt)("a",{parentName:"p",href:"transforms_tutorial.html"},"Transforms")," ||\n",(0,A.kt)("a",{parentName:"p",href:"buildmodel_tutorial.html"},"Build Model")," ||\n",(0,A.kt)("a",{parentName:"p",href:"autogradqs_tutorial.html"},"Autograd")," ||\n",(0,A.kt)("a",{parentName:"p",href:"optimization_tutorial.html"},"Optimization")," ||\n",(0,A.kt)("a",{parentName:"p",href:"saveloadrun_tutorial.html"},"Save & Load Model")),(0,A.kt)("h1",{id:"datasets--dataloaders"},"Datasets & DataLoaders"),(0,A.kt)("p",null,"Code for processing data samples can get messy and hard to maintain; we ideally want our dataset code\nto be decoupled from our model training code for better readability and modularity.\nPyTorch provides two data primitives: ",(0,A.kt)("inlineCode",{parentName:"p"},"torch.utils.data.DataLoader")," and ",(0,A.kt)("inlineCode",{parentName:"p"},"torch.utils.data.Dataset"),"\nthat allow you to use pre-loaded datasets as well as your own data.\n",(0,A.kt)("inlineCode",{parentName:"p"},"Dataset")," stores the samples and their corresponding labels, and ",(0,A.kt)("inlineCode",{parentName:"p"},"DataLoader")," wraps an iterable around\nthe ",(0,A.kt)("inlineCode",{parentName:"p"},"Dataset")," to enable easy access to the samples."),(0,A.kt)("p",null,"PyTorch domain libraries provide a number of pre-loaded datasets (such as FashionMNIST) that\nsubclass ",(0,A.kt)("inlineCode",{parentName:"p"},"torch.utils.data.Dataset")," and implement functions specific to the particular data.\nThey can be used to prototype and benchmark your model. You can find them\nhere: ",(0,A.kt)("a",{parentName:"p",href:"https://pytorch.org/vision/stable/datasets.html"},"Image Datasets"),",\n",(0,A.kt)("a",{parentName:"p",href:"https://pytorch.org/text/stable/datasets.html"},"Text Datasets"),", and\n",(0,A.kt)("a",{parentName:"p",href:"https://pytorch.org/audio/stable/datasets.html"},"Audio Datasets")),(0,A.kt)("h2",{id:"loading-a-dataset"},"Loading a Dataset"),(0,A.kt)("p",null,"Here is an example of how to load the ",(0,A.kt)("a",{parentName:"p",href:"https://research.zalando.com/project/fashion_mnist/fashion_mnist/"},"Fashion-MNIST")," dataset from TorchVision.\nFashion-MNIST is a dataset of Zalando\u2019s article images consisting of 60,000 training examples and 10,000 test examples.\nEach example comprises a 28\xd728 grayscale image and an associated label from one of 10 classes."),(0,A.kt)("p",null,"We load the ",(0,A.kt)("a",{parentName:"p",href:"https://pytorch.org/vision/stable/datasets.html#fashion-mnist"},"FashionMNIST Dataset")," with the following parameters:"),(0,A.kt)("ul",null,(0,A.kt)("li",{parentName:"ul"},(0,A.kt)("inlineCode",{parentName:"li"},"root")," is the path where the train/test data is stored,"),(0,A.kt)("li",{parentName:"ul"},(0,A.kt)("inlineCode",{parentName:"li"},"train")," specifies training or test dataset,"),(0,A.kt)("li",{parentName:"ul"},(0,A.kt)("inlineCode",{parentName:"li"},"download=True")," downloads the data from the internet if it's not available at ",(0,A.kt)("inlineCode",{parentName:"li"},"root"),"."),(0,A.kt)("li",{parentName:"ul"},(0,A.kt)("inlineCode",{parentName:"li"},"transform")," and ",(0,A.kt)("inlineCode",{parentName:"li"},"target_transform")," specify the feature and label transformations")),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre",className:"language-python"},'%matplotlib inline\n\nimport torch\nfrom torch.utils.data import Dataset\nfrom torchvision import datasets\nfrom torchvision.transforms import ToTensor\nimport matplotlib.pyplot as plt\n\n\ntraining_data = datasets.FashionMNIST(\n    root="data",\n    train=True,\n    download=True,\n    transform=ToTensor()\n)\n\ntest_data = datasets.FashionMNIST(\n    root="data",\n    train=False,\n    download=True,\n    transform=ToTensor()\n)\n')),(0,A.kt)("h2",{id:"iterating-and-visualizing-the-dataset"},"Iterating and Visualizing the Dataset"),(0,A.kt)("p",null,"We can index ",(0,A.kt)("inlineCode",{parentName:"p"},"Datasets")," manually like a list: ",(0,A.kt)("inlineCode",{parentName:"p"},"training_data[index]"),".\nWe use ",(0,A.kt)("inlineCode",{parentName:"p"},"matplotlib")," to visualize some samples in our training data."),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre",className:"language-python"},'labels_map = {\n    0: "T-Shirt",\n    1: "Trouser",\n    2: "Pullover",\n    3: "Dress",\n    4: "Coat",\n    5: "Sandal",\n    6: "Shirt",\n    7: "Sneaker",\n    8: "Bag",\n    9: "Ankle Boot",\n}\nfigure = plt.figure(figsize=(8, 8))\ncols, rows = 3, 3\nfor i in range(1, cols * rows + 1):\n    sample_idx = torch.randint(len(training_data), size=(1,)).item()\n    img, label = training_data[sample_idx]\n    figure.add_subplot(rows, cols, i)\n    plt.title(labels_map[label])\n    plt.axis("off")\n    plt.imshow(img.squeeze(), cmap="gray")\nplt.show()\n')),(0,A.kt)("p",null,(0,A.kt)("img",{alt:"png",src:a(5620).Z,width:"638",height:"658"})),(0,A.kt)("p",null,"..\n.. figure:: /_static/img/basics/fashion_mnist.png\n:alt: fashion_mnist"),(0,A.kt)("hr",null),(0,A.kt)("h2",{id:"creating-a-custom-dataset-for-your-files"},"Creating a Custom Dataset for your files"),(0,A.kt)("p",null,"A custom Dataset class must implement three functions: ",(0,A.kt)("inlineCode",{parentName:"p"},"__init__"),", ",(0,A.kt)("inlineCode",{parentName:"p"},"__len__"),", and ",(0,A.kt)("inlineCode",{parentName:"p"},"__getitem__"),".\nTake a look at this implementation; the FashionMNIST images are stored\nin a directory ",(0,A.kt)("inlineCode",{parentName:"p"},"img_dir"),", and their labels are stored separately in a CSV file ",(0,A.kt)("inlineCode",{parentName:"p"},"annotations_file"),"."),(0,A.kt)("p",null,"In the next sections, we'll break down what's happening in each of these functions."),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre",className:"language-python"},"import os\nimport pandas as pd\nfrom torchvision.io import read_image\n\nclass CustomImageDataset(Dataset):\n    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):\n        self.img_labels = pd.read_csv(annotations_file)\n        self.img_dir = img_dir\n        self.transform = transform\n        self.target_transform = target_transform\n\n    def __len__(self):\n        return len(self.img_labels)\n\n    def __getitem__(self, idx):\n        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])\n        image = read_image(img_path)\n        label = self.img_labels.iloc[idx, 1]\n        if self.transform:\n            image = self.transform(image)\n        if self.target_transform:\n            label = self.target_transform(label)\n        return image, label\n")),(0,A.kt)("h3",{id:"init"},(0,A.kt)("strong",{parentName:"h3"},"init")),(0,A.kt)("p",null,"The ",(0,A.kt)("strong",{parentName:"p"},"init")," function is run once when instantiating the Dataset object. We initialize\nthe directory containing the images, the annotations file, and both transforms (covered\nin more detail in the next section)."),(0,A.kt)("p",null,"The labels.csv file looks like: ::"),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre"},"tshirt1.jpg, 0\ntshirt2.jpg, 0\n......\nankleboot999.jpg, 9\n")),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre",className:"language-python"},"def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):\n    self.img_labels = pd.read_csv(annotations_file)\n    self.img_dir = img_dir\n    self.transform = transform\n    self.target_transform = target_transform\n")),(0,A.kt)("h3",{id:"len"},(0,A.kt)("strong",{parentName:"h3"},"len")),(0,A.kt)("p",null,"The ",(0,A.kt)("strong",{parentName:"p"},"len")," function returns the number of samples in our dataset."),(0,A.kt)("p",null,"Example:"),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre",className:"language-python"},"def __len__(self):\n    return len(self.img_labels)\n")),(0,A.kt)("h3",{id:"getitem"},(0,A.kt)("strong",{parentName:"h3"},"getitem")),(0,A.kt)("p",null,"The ",(0,A.kt)("strong",{parentName:"p"},"getitem")," function loads and returns a sample from the dataset at the given index ",(0,A.kt)("inlineCode",{parentName:"p"},"idx"),".\nBased on the index, it identifies the image's location on disk, converts that to a tensor using ",(0,A.kt)("inlineCode",{parentName:"p"},"read_image"),", retrieves the\ncorresponding label from the csv data in ",(0,A.kt)("inlineCode",{parentName:"p"},"self.img_labels"),", calls the transform functions on them (if applicable), and returns the\ntensor image and corresponding label in a tuple."),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre",className:"language-python"},"def __getitem__(self, idx):\n    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])\n    image = read_image(img_path)\n    label = self.img_labels.iloc[idx, 1]\n    if self.transform:\n        image = self.transform(image)\n    if self.target_transform:\n        label = self.target_transform(label)\n    return image, label\n")),(0,A.kt)("hr",null),(0,A.kt)("h2",{id:"preparing-your-data-for-training-with-dataloaders"},"Preparing your data for training with DataLoaders"),(0,A.kt)("p",null,"The ",(0,A.kt)("inlineCode",{parentName:"p"},"Dataset")," retrieves our dataset's features and labels one sample at a time. While training a model, we typically want to\npass samples in \"minibatches\", reshuffle the data at every epoch to reduce model overfitting, and use Python's ",(0,A.kt)("inlineCode",{parentName:"p"},"multiprocessing")," to\nspeed up data retrieval."),(0,A.kt)("p",null,(0,A.kt)("inlineCode",{parentName:"p"},"DataLoader")," is an iterable that abstracts this complexity for us in an easy API."),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre",className:"language-python"},"from torch.utils.data import DataLoader\n\ntrain_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)\ntest_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)\n")),(0,A.kt)("h2",{id:"iterate-through-the-dataloader"},"Iterate through the DataLoader"),(0,A.kt)("p",null,"We have loaded that dataset into the ",(0,A.kt)("inlineCode",{parentName:"p"},"DataLoader")," and can iterate through the dataset as needed.\nEach iteration below returns a batch of ",(0,A.kt)("inlineCode",{parentName:"p"},"train_features")," and ",(0,A.kt)("inlineCode",{parentName:"p"},"train_labels")," (containing ",(0,A.kt)("inlineCode",{parentName:"p"},"batch_size=64")," features and labels respectively).\nBecause we specified ",(0,A.kt)("inlineCode",{parentName:"p"},"shuffle=True"),", after we iterate over all batches the data is shuffled (for finer-grained control over\nthe data loading order, take a look at ",(0,A.kt)("a",{parentName:"p",href:"https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler"},"Samplers"),")."),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre",className:"language-python"},'# Display image and label.\ntrain_features, train_labels = next(iter(train_dataloader))\nprint(f"Feature batch shape: {train_features.size()}")\nprint(f"Labels batch shape: {train_labels.size()}")\nimg = train_features[0].squeeze()\nlabel = train_labels[0]\nplt.imshow(img, cmap="gray")\nplt.show()\nprint(f"Label: {label}")\n')),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre"},"Feature batch shape: torch.Size([64, 1, 28, 28])\nLabels batch shape: torch.Size([64])\n")),(0,A.kt)("p",null,(0,A.kt)("img",{alt:"png",src:a(6540).Z,width:"416",height:"413"})),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre"},"Label: 5\n")),(0,A.kt)("hr",null),(0,A.kt)("h2",{id:"further-reading"},"Further Reading"),(0,A.kt)("ul",null,(0,A.kt)("li",{parentName:"ul"},(0,A.kt)("a",{parentName:"li",href:"https://pytorch.org/docs/stable/data.html"},"torch.utils.data API"))))}m.isMDXComponent=!0},6540:(e,t,a)=>{a.d(t,{Z:()=>n});const n="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAaAAAAGdCAYAAABU0qcqAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAc1UlEQVR4nO3df2xV9f3H8ddtoZcC7a2ltLdXftjiD1R+GFG6TmUoDbQuRJQl4tyCzmhwxQyZunSbovuRbixzxoXpYhaYmagzGRD9g0WLLXG2GFBC0FkpqVICbbVL7y3FFmg/3z/69W5Xfp7DvX235flIPgn3nvPueffT0/vi9J5+GnDOOQEAMMjSrBsAAFyYCCAAgAkCCABgggACAJgggAAAJgggAIAJAggAYIIAAgCYGGXdwNf19/fr0KFDysrKUiAQsG4HAOCRc05dXV2KRCJKSzv9dc6QC6BDhw5p8uTJ1m0AAM5TS0uLJk2adNrtQ+5HcFlZWdYtAACS4Gyv5ykLoHXr1umSSy7RmDFjVFJSovfee++c6vixGwCMDGd7PU9JAL366qtavXq11qxZo/fff1+zZ8/WokWL1N7enorDAQCGI5cCc+fOdZWVlfHHfX19LhKJuOrq6rPWRqNRJ4nBYDAYw3xEo9Ezvt4n/Qro2LFj2rVrl8rKyuLPpaWlqaysTPX19Sft39vbq1gsljAAACNf0gPoiy++UF9fnwoKChKeLygoUGtr60n7V1dXKxQKxQd3wAHAhcH8LriqqipFo9H4aGlpsW4JADAIkv57QHl5eUpPT1dbW1vC821tbQqHwyftHwwGFQwGk90GAGCIS/oVUEZGhubMmaOampr4c/39/aqpqVFpaWmyDwcAGKZSshLC6tWrtXz5cl133XWaO3eunnnmGXV3d+vee+9NxeEAAMNQSgLozjvv1Oeff64nnnhCra2tuuaaa7R169aTbkwAAFy4As45Z93E/4rFYgqFQtZtAADOUzQaVXZ29mm3m98FBwC4MBFAAAATBBAAwAQBBAAwQQABAEwQQAAAEwQQAMAEAQQAMEEAAQBMEEAAABMEEADABAEEADBBAAEATBBAAAATBBAAwAQBBAAwQQABAEwQQAAAEwQQAMAEAQQAMEEAAQBMEEAAABMEEADABAEEADBBAAEATBBAAAATBBAAwAQBBAAwQQABAEwQQAAAEwQQAMAEAQQAMEEAAQBMEEAAABMEEADABAEEADBBAAEATBBAAAATBBAAwAQBBAAwQQABAEwQQAAAEwQQAMAEAQQAMEEAAQBMEEAAABMEEADABAEEADBBAAEATBBAAAATBBAAwAQBBAAwQQABAEwQQAAAEwQQAMBE0gPoySefVCAQSBjTp09P9mEAAMPcqFR80KuvvlpvvfXWfw8yKiWHAQAMYylJhlGjRikcDqfiQwMARoiUvAe0b98+RSIRFRcX6+6779aBAwdOu29vb69isVjCAACMfEkPoJKSEm3YsEFbt27Vc889p+bmZt10003q6uo65f7V1dUKhULxMXny5GS3BAAYggLOOZfKA3R2dmrq1Kl6+umndd999520vbe3V729vfHHsViMEAKAESAajSo7O/u021N+d0BOTo4uv/xyNTU1nXJ7MBhUMBhMdRsAgCEm5b8HdOTIEe3fv1+FhYWpPhQAYBhJegA98sgjqqur06effqp3331Xt99+u9LT03XXXXcl+1AAgGEs6T+CO3jwoO666y51dHRo4sSJuvHGG9XQ0KCJEycm+1AAgGEs5TcheBWLxRQKhazbAOCRn/dyV65c6bnmmmuu8Vyzfft2zzWS9MILL/iqw4Cz3YTAWnAAABMEEADABAEEADBBAAEATBBAAAATBBAAwAQBBAAwQQABAEwQQAAAEwQQAMAEAQQAMEEAAQBMpPwP0gGwc9FFF/mqy8zM9FxTVVXlucbP3wnr6OjwXHPrrbd6rpGkiy++2HONn8WUs7KyPNdMmjTJc40klZeX+6pLBa6AAAAmCCAAgAkCCABgggACAJgggAAAJgggAIAJAggAYIIAAgCYIIAAACYIIACACQIIAGCCAAIAmCCAAAAmWA0bQ14gEBi0YznnPNf46c9PTX9/v+eaZcuWea6RpGuvvdZzTXt7u+eaq666ynNNLBbzXPPJJ594rpH8rSbuZ2VrPyt8X3nllZ5rJOmb3/ym55p3333X17HOhisgAIAJAggAYIIAAgCYIIAAACYIIACACQIIAGCCAAIAmCCAAAAmCCAAgAkCCABgggACAJgggAAAJliMFIPKzyKcfhYIxYCcnBxfdT09PZ5r0tPTPdf09fV5rhk1yvvLVmFhoecaSRo7dqznGj/n64cffui55qOPPvJcI0mdnZ2+6lKBKyAAgAkCCABgggACAJgg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"},5620:(e,t,a)=>{a.d(t,{Z:()=>n});const n=a.p+"assets/images/04-Data_5_0-21b7a5d9935085ac88f196f93bde2d94.png"}}]);