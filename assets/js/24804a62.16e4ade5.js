"use strict";(self.webpackChunk=self.webpackChunk||[]).push([[576],{3905:(e,t,a)=>{a.d(t,{Zo:()=>d,kt:()=>f});var n=a(7294);function A(e,t,a){return t in e?Object.defineProperty(e,t,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[t]=a,e}function r(e,t){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),a.push.apply(a,n)}return a}function i(e){for(var t=1;t<arguments.length;t++){var a=null!=arguments[t]?arguments[t]:{};t%2?r(Object(a),!0).forEach((function(t){A(e,t,a[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):r(Object(a)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(a,t))}))}return e}function s(e,t){if(null==e)return{};var a,n,A=function(e,t){if(null==e)return{};var a,n,A={},r=Object.keys(e);for(n=0;n<r.length;n++)a=r[n],t.indexOf(a)>=0||(A[a]=e[a]);return A}(e,t);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);for(n=0;n<r.length;n++)a=r[n],t.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(A[a]=e[a])}return A}var l=n.createContext({}),o=function(e){var t=n.useContext(l),a=t;return e&&(a="function"==typeof e?e(t):i(i({},t),e)),a},d=function(e){var t=o(e.components);return n.createElement(l.Provider,{value:t},e.children)},p="mdxType",m={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},u=n.forwardRef((function(e,t){var a=e.components,A=e.mdxType,r=e.originalType,l=e.parentName,d=s(e,["components","mdxType","originalType","parentName"]),p=o(a),u=A,f=p["".concat(l,".").concat(u)]||p[u]||m[u]||r;return a?n.createElement(f,i(i({ref:t},d),{},{components:a})):n.createElement(f,i({ref:t},d))}));function f(e,t){var a=arguments,A=t&&t.mdxType;if("string"==typeof e||A){var r=a.length,i=new Array(r);i[0]=u;var s={};for(var l in t)hasOwnProperty.call(t,l)&&(s[l]=t[l]);s.originalType=e,s[p]="string"==typeof e?e:A,i[1]=s;for(var o=2;o<r;o++)i[o]=a[o];return n.createElement.apply(null,i)}return n.createElement.apply(null,a)}u.displayName="MDXCreateElement"},5138:(e,t,a)=>{a.r(t),a.d(t,{assets:()=>l,contentTitle:()=>i,default:()=>m,frontMatter:()=>r,metadata:()=>s,toc:()=>o});var n=a(7462),A=(a(7294),a(3905));const r={},i=void 0,s={unversionedId:"Data",id:"Data",title:"Data",description:"Learn the Basics ||",source:"@site/docs/04-Data.md",sourceDirName:".",slug:"/Data",permalink:"/pytorch-basics/docs/Data",draft:!1,tags:[],version:"current",sidebarPosition:4,frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Tensors",permalink:"/pytorch-basics/docs/Tensors"},next:{title:"Transforms",permalink:"/pytorch-basics/docs/Transforms"}},l={},o=[{value:"Loading a Dataset",id:"loading-a-dataset",level:2},{value:"Iterating and Visualizing the Dataset",id:"iterating-and-visualizing-the-dataset",level:2},{value:"Creating a Custom Dataset for your files",id:"creating-a-custom-dataset-for-your-files",level:2},{value:"<strong>init</strong>",id:"init",level:3},{value:"<strong>len</strong>",id:"len",level:3},{value:"<strong>getitem</strong>",id:"getitem",level:3},{value:"Preparing your data for training with DataLoaders",id:"preparing-your-data-for-training-with-dataloaders",level:2},{value:"Iterate through the DataLoader",id:"iterate-through-the-dataloader",level:2},{value:"Further Reading",id:"further-reading",level:2}],d={toc:o},p="wrapper";function m(e){let{components:t,...r}=e;return(0,A.kt)(p,(0,n.Z)({},d,r,{components:t,mdxType:"MDXLayout"}),(0,A.kt)("p",null,(0,A.kt)("a",{parentName:"p",href:"intro.html"},"Learn the Basics")," ||\n",(0,A.kt)("a",{parentName:"p",href:"quickstart_tutorial.html"},"Quickstart")," ||\n",(0,A.kt)("a",{parentName:"p",href:"tensorqs_tutorial.html"},"Tensors")," ||\n",(0,A.kt)("strong",{parentName:"p"},"Datasets & DataLoaders")," ||\n",(0,A.kt)("a",{parentName:"p",href:"transforms_tutorial.html"},"Transforms")," ||\n",(0,A.kt)("a",{parentName:"p",href:"buildmodel_tutorial.html"},"Build Model")," ||\n",(0,A.kt)("a",{parentName:"p",href:"autogradqs_tutorial.html"},"Autograd")," ||\n",(0,A.kt)("a",{parentName:"p",href:"optimization_tutorial.html"},"Optimization")," ||\n",(0,A.kt)("a",{parentName:"p",href:"saveloadrun_tutorial.html"},"Save & Load Model")),(0,A.kt)("h1",{id:"datasets--dataloaders"},"Datasets & DataLoaders"),(0,A.kt)("p",null,"Code for processing data samples can get messy and hard to maintain; we ideally want our dataset code\nto be decoupled from our model training code for better readability and modularity.\nPyTorch provides two data primitives: ",(0,A.kt)("inlineCode",{parentName:"p"},"torch.utils.data.DataLoader")," and ",(0,A.kt)("inlineCode",{parentName:"p"},"torch.utils.data.Dataset"),"\nthat allow you to use pre-loaded datasets as well as your own data.\n",(0,A.kt)("inlineCode",{parentName:"p"},"Dataset")," stores the samples and their corresponding labels, and ",(0,A.kt)("inlineCode",{parentName:"p"},"DataLoader")," wraps an iterable around\nthe ",(0,A.kt)("inlineCode",{parentName:"p"},"Dataset")," to enable easy access to the samples."),(0,A.kt)("p",null,"PyTorch domain libraries provide a number of pre-loaded datasets (such as FashionMNIST) that\nsubclass ",(0,A.kt)("inlineCode",{parentName:"p"},"torch.utils.data.Dataset")," and implement functions specific to the particular data.\nThey can be used to prototype and benchmark your model. You can find them\nhere: ",(0,A.kt)("a",{parentName:"p",href:"https://pytorch.org/vision/stable/datasets.html"},"Image Datasets"),",\n",(0,A.kt)("a",{parentName:"p",href:"https://pytorch.org/text/stable/datasets.html"},"Text Datasets"),", and\n",(0,A.kt)("a",{parentName:"p",href:"https://pytorch.org/audio/stable/datasets.html"},"Audio Datasets")),(0,A.kt)("h2",{id:"loading-a-dataset"},"Loading a Dataset"),(0,A.kt)("p",null,"Here is an example of how to load the ",(0,A.kt)("a",{parentName:"p",href:"https://research.zalando.com/project/fashion_mnist/fashion_mnist/"},"Fashion-MNIST")," dataset from TorchVision.\nFashion-MNIST is a dataset of Zalando\u2019s article images consisting of 60,000 training examples and 10,000 test examples.\nEach example comprises a 28\xd728 grayscale image and an associated label from one of 10 classes."),(0,A.kt)("p",null,"We load the ",(0,A.kt)("a",{parentName:"p",href:"https://pytorch.org/vision/stable/datasets.html#fashion-mnist"},"FashionMNIST Dataset")," with the following parameters:"),(0,A.kt)("ul",null,(0,A.kt)("li",{parentName:"ul"},(0,A.kt)("inlineCode",{parentName:"li"},"root")," is the path where the train/test data is stored,"),(0,A.kt)("li",{parentName:"ul"},(0,A.kt)("inlineCode",{parentName:"li"},"train")," specifies training or test dataset,"),(0,A.kt)("li",{parentName:"ul"},(0,A.kt)("inlineCode",{parentName:"li"},"download=True")," downloads the data from the internet if it's not available at ",(0,A.kt)("inlineCode",{parentName:"li"},"root"),"."),(0,A.kt)("li",{parentName:"ul"},(0,A.kt)("inlineCode",{parentName:"li"},"transform")," and ",(0,A.kt)("inlineCode",{parentName:"li"},"target_transform")," specify the feature and label transformations")),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre",className:"language-python"},'%matplotlib inline\n\nimport torch\nfrom torch.utils.data import Dataset\nfrom torchvision import datasets\nfrom torchvision.transforms import ToTensor\nimport matplotlib.pyplot as plt\n\n\ntraining_data = datasets.FashionMNIST(\n    root="data",\n    train=True,\n    download=True,\n    transform=ToTensor()\n)\n\ntest_data = datasets.FashionMNIST(\n    root="data",\n    train=False,\n    download=True,\n    transform=ToTensor()\n)\n')),(0,A.kt)("h2",{id:"iterating-and-visualizing-the-dataset"},"Iterating and Visualizing the Dataset"),(0,A.kt)("p",null,"We can index ",(0,A.kt)("inlineCode",{parentName:"p"},"Datasets")," manually like a list: ",(0,A.kt)("inlineCode",{parentName:"p"},"training_data[index]"),".\nWe use ",(0,A.kt)("inlineCode",{parentName:"p"},"matplotlib")," to visualize some samples in our training data."),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre",className:"language-python"},'labels_map = {\n    0: "T-Shirt",\n    1: "Trouser",\n    2: "Pullover",\n    3: "Dress",\n    4: "Coat",\n    5: "Sandal",\n    6: "Shirt",\n    7: "Sneaker",\n    8: "Bag",\n    9: "Ankle Boot",\n}\nfigure = plt.figure(figsize=(8, 8))\ncols, rows = 3, 3\nfor i in range(1, cols * rows + 1):\n    sample_idx = torch.randint(len(training_data), size=(1,)).item()\n    img, label = training_data[sample_idx]\n    figure.add_subplot(rows, cols, i)\n    plt.title(labels_map[label])\n    plt.axis("off")\n    plt.imshow(img.squeeze(), cmap="gray")\nplt.show()\n')),(0,A.kt)("p",null,(0,A.kt)("img",{alt:"png",src:a(5620).Z,width:"638",height:"658"})),(0,A.kt)("p",null,"..\n.. figure:: /_static/img/basics/fashion_mnist.png\n:alt: fashion_mnist"),(0,A.kt)("hr",null),(0,A.kt)("h2",{id:"creating-a-custom-dataset-for-your-files"},"Creating a Custom Dataset for your files"),(0,A.kt)("p",null,"A custom Dataset class must implement three functions: ",(0,A.kt)("inlineCode",{parentName:"p"},"__init__"),", ",(0,A.kt)("inlineCode",{parentName:"p"},"__len__"),", and ",(0,A.kt)("inlineCode",{parentName:"p"},"__getitem__"),".\nTake a look at this implementation; the FashionMNIST images are stored\nin a directory ",(0,A.kt)("inlineCode",{parentName:"p"},"img_dir"),", and their labels are stored separately in a CSV file ",(0,A.kt)("inlineCode",{parentName:"p"},"annotations_file"),"."),(0,A.kt)("p",null,"In the next sections, we'll break down what's happening in each of these functions."),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre",className:"language-python"},"import os\nimport pandas as pd\nfrom torchvision.io import read_image\n\nclass CustomImageDataset(Dataset):\n    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):\n        self.img_labels = pd.read_csv(annotations_file)\n        self.img_dir = img_dir\n        self.transform = transform\n        self.target_transform = target_transform\n\n    def __len__(self):\n        return len(self.img_labels)\n\n    def __getitem__(self, idx):\n        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])\n        image = read_image(img_path)\n        label = self.img_labels.iloc[idx, 1]\n        if self.transform:\n            image = self.transform(image)\n        if self.target_transform:\n            label = self.target_transform(label)\n        return image, label\n")),(0,A.kt)("h3",{id:"init"},(0,A.kt)("strong",{parentName:"h3"},"init")),(0,A.kt)("p",null,"The ",(0,A.kt)("strong",{parentName:"p"},"init")," function is run once when instantiating the Dataset object. We initialize\nthe directory containing the images, the annotations file, and both transforms (covered\nin more detail in the next section)."),(0,A.kt)("p",null,"The labels.csv file looks like: ::"),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre"},"tshirt1.jpg, 0\ntshirt2.jpg, 0\n......\nankleboot999.jpg, 9\n")),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre",className:"language-python"},"def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):\n    self.img_labels = pd.read_csv(annotations_file)\n    self.img_dir = img_dir\n    self.transform = transform\n    self.target_transform = target_transform\n")),(0,A.kt)("h3",{id:"len"},(0,A.kt)("strong",{parentName:"h3"},"len")),(0,A.kt)("p",null,"The ",(0,A.kt)("strong",{parentName:"p"},"len")," function returns the number of samples in our dataset."),(0,A.kt)("p",null,"Example:"),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre",className:"language-python"},"def __len__(self):\n    return len(self.img_labels)\n")),(0,A.kt)("h3",{id:"getitem"},(0,A.kt)("strong",{parentName:"h3"},"getitem")),(0,A.kt)("p",null,"The ",(0,A.kt)("strong",{parentName:"p"},"getitem")," function loads and returns a sample from the dataset at the given index ",(0,A.kt)("inlineCode",{parentName:"p"},"idx"),".\nBased on the index, it identifies the image's location on disk, converts that to a tensor using ",(0,A.kt)("inlineCode",{parentName:"p"},"read_image"),", retrieves the\ncorresponding label from the csv data in ",(0,A.kt)("inlineCode",{parentName:"p"},"self.img_labels"),", calls the transform functions on them (if applicable), and returns the\ntensor image and corresponding label in a tuple."),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre",className:"language-python"},"def __getitem__(self, idx):\n    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])\n    image = read_image(img_path)\n    label = self.img_labels.iloc[idx, 1]\n    if self.transform:\n        image = self.transform(image)\n    if self.target_transform:\n        label = self.target_transform(label)\n    return image, label\n")),(0,A.kt)("hr",null),(0,A.kt)("h2",{id:"preparing-your-data-for-training-with-dataloaders"},"Preparing your data for training with DataLoaders"),(0,A.kt)("p",null,"The ",(0,A.kt)("inlineCode",{parentName:"p"},"Dataset")," retrieves our dataset's features and labels one sample at a time. While training a model, we typically want to\npass samples in \"minibatches\", reshuffle the data at every epoch to reduce model overfitting, and use Python's ",(0,A.kt)("inlineCode",{parentName:"p"},"multiprocessing")," to\nspeed up data retrieval."),(0,A.kt)("p",null,(0,A.kt)("inlineCode",{parentName:"p"},"DataLoader")," is an iterable that abstracts this complexity for us in an easy API."),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre",className:"language-python"},"from torch.utils.data import DataLoader\n\ntrain_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)\ntest_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)\n")),(0,A.kt)("h2",{id:"iterate-through-the-dataloader"},"Iterate through the DataLoader"),(0,A.kt)("p",null,"We have loaded that dataset into the ",(0,A.kt)("inlineCode",{parentName:"p"},"DataLoader")," and can iterate through the dataset as needed.\nEach iteration below returns a batch of ",(0,A.kt)("inlineCode",{parentName:"p"},"train_features")," and ",(0,A.kt)("inlineCode",{parentName:"p"},"train_labels")," (containing ",(0,A.kt)("inlineCode",{parentName:"p"},"batch_size=64")," features and labels respectively).\nBecause we specified ",(0,A.kt)("inlineCode",{parentName:"p"},"shuffle=True"),", after we iterate over all batches the data is shuffled (for finer-grained control over\nthe data loading order, take a look at ",(0,A.kt)("a",{parentName:"p",href:"https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler"},"Samplers"),")."),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre",className:"language-python"},'# Display image and label.\ntrain_features, train_labels = next(iter(train_dataloader))\nprint(f"Feature batch shape: {train_features.size()}")\nprint(f"Labels batch shape: {train_labels.size()}")\nimg = train_features[0].squeeze()\nlabel = train_labels[0]\nplt.imshow(img, cmap="gray")\nplt.show()\nprint(f"Label: {label}")\n')),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre"},"Feature batch shape: torch.Size([64, 1, 28, 28])\nLabels batch shape: torch.Size([64])\n")),(0,A.kt)("p",null,(0,A.kt)("img",{alt:"png",src:a(6540).Z,width:"416",height:"413"})),(0,A.kt)("pre",null,(0,A.kt)("code",{parentName:"pre"},"Label: 7\n")),(0,A.kt)("hr",null),(0,A.kt)("h2",{id:"further-reading"},"Further Reading"),(0,A.kt)("ul",null,(0,A.kt)("li",{parentName:"ul"},(0,A.kt)("a",{parentName:"li",href:"https://pytorch.org/docs/stable/data.html"},"torch.utils.data API"))))}m.isMDXComponent=!0},6540:(e,t,a)=>{a.d(t,{Z:()=>n});const n="data:image/png;base64,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AQBMEEAAABMEEADABAEEADBBAAEATDCMdICEQiHvmhEj/L883d3d3jWDXVlZmXfN6tWrA+1r+vTp3jXNzc3eNa+99tqA7OeKK67wrpGk0tJS75qenh7vmiDn64kTJ7xrggz7lKTe3l7vGuecd02Qf+ujRo3yrpGCDYBNFq6AAAAmCCAAgAkCCABgggACAJgggAAAJgggAIAJAggAYIIAAgCYIIAAACYIIACACQIIAGCCAAIAmBg2w0iDDPsMMjQwqCD7GuyDRb///e971zz//PPeNZ9//rl3zUcffeRdI0kvvviid017e7t3TVpamnfNVVdd5V0zdepU7xpJikaj3jXZ2dneNampqd41x48f964JMsBUCvZ9JcjXtqOjw7vmkksu8a6RpM8++yxQXTJwBQQAMEEAAQBMeAfQzp07dfPNN6ugoEChUEhbt26Ne/6uu+5SKBSKW4sXL05UvwCAYcI7gDo7O1VcXKz169f3u83ixYvV0tLSt1555ZULahIAMPx434SwZMkSLVmy5JzbhMNh5eXlBW4KADD8JeU9oJqaGuXk5GjGjBm67777dPTo0X637erqUiwWi1sAgOEv4QG0ePFivfTSS6qurtavf/1r1dbWasmSJf3+vvjKykpFIpG+NWnSpES3BAAYhBL+c0C33357359nz56tOXPmaOrUqaqpqdGCBQvO2H7NmjVavXp138exWIwQAoCLQNJvwy4qKlJ2drYaGhrO+nw4HFZGRkbcAgAMf0kPoEOHDuno0aPKz89P9q4AAEOI90twx44di7uaaWpq0r59+5SVlaWsrCw99dRTWrZsmfLy8tTY2KhHHnlE06ZN06JFixLaOABgaPMOoD179uiGG27o+/jL929WrFihDRs2aP/+/frDH/6g9vZ2FRQUaOHChfrFL36hcDicuK4BAEOedwDNnz//nIM133rrrQtqKKiBHCw6UIIMn/zud7/rXVNUVORdIynQz3pt27bNu2bHjh3eNZ988ol3jSSlpPi/Kh3kppl58+Z510yfPt27pr+7T89n9OjR3jVBB376CvI1CjIgVFKgHws514+d9CfIe99BhxVnZWUFqksGZsEBAEwQQAAAEwQQAMAEAQQAMEEAAQBMEEAAABMEEADABAEEADBBAAEATBBAAAATBBAAwAQBBAAwQQABAEwk/FdyW7nuuuu8ayZMmBBo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SdLgOh+s74L4Ol599VUXDofdpk2b3D/+8Q937733uszMTNfa2mrd2oD66U9/6mpqalxTU5P7+9//7srLy112drY7cuSIdWtJ1dHR4T744AP3wQcfOEnumWeecR988IH797//7Zxz7le/+pXLzMx027Ztc/v373e33HKLKywsdCdOnDDuPLHOdRw6OjrcQw895Orq6lxTU5N7++233dVXX+0uv/xyd/LkSevWE+a+++5zkUjE1dTUuJaWlr51/Pjxvm1WrlzpJk+e7Hbs2OH27NnjSktLXWlpqWHXiXe+49DQ0OCefvppt2fPHtfU1OS2bdvmioqKXFlZmXHn8YZEADnn3O9+9zs3efJkl5aW5ubNm+d27dpl3dKAW758ucvPz3dpaWnu0ksvdcuXL3cNDQ3WbSXdO++84ySdsVasWOGcO30r9uOPP+5yc3NdOBx2CxYscPX19bZNJ8G5jsPx48fdwoUL3YQJE9zIkSPdlClT3D333DPs/pN2tr+/JLdx48a+bU6cOOF+/OMfu0suucSNGTPG3Xrrra6lpcWu6SQ433E4ePCgKysrc1lZWS4cDrtp06a5hx9+2EWjUdvGv4JfxwAAMDHo3wMCAAxPBBAAwAQBBAAwQQABAEwQQAAAEwQQAMAEAQQAMEEAAQBMEEAAABMEEADABAEEADBBAAEATPw/+fyLznrUdk8AAAAASUVORK5CYII="},5620:(e,t,a)=>{a.d(t,{Z:()=>n});const n=a.p+"assets/images/04-Data_5_0-a7dfbd377f7dcde23ab2468dfb616fc2.png"}}]);