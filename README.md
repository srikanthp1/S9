# Image classification - CIFAR add.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/), 
[PyTorch](https://pytorch.org/), 
[torchvision](https://github.com/pytorch/vision) 0.8, 
Uses [matplotlib](https://matplotlib.org/)  for ploting accuracy and losses.

## Info

 * we are training our model on CIFAR dataset. 
 * we are using a custom convolutional neural network (CNN) architectures which includes dilation, depthwise seperable convs, strided convs 
 * Implemented in pytorch 

## About

* transforms.py contains the transforms we used using Albumentation library
* dataset.py has dataset class for applying transforms 
* model.py has model classes and other related functions 
* utils.py has some graph functions and others 

## Results 

### Train accuracy 

* after 60 epochs 

* 79.29%

### Test accuracy 

* 85.59%

## Usage

```bash
git clone https://github.com/srikanthp1/S9.git
```
* utils.py has util functions
* model.py has models 
* run cell by cell to download, visualize data and train model


## Model details

```python
model = Net().to(device)
summary(model, input_size=(3, 32, 32))
```

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 44, 32, 32]           1,188
              ReLU-2           [-1, 44, 32, 32]               0
       BatchNorm2d-3           [-1, 44, 32, 32]              88
           Dropout-4           [-1, 44, 32, 32]               0
            Conv2d-5           [-1, 44, 32, 32]          17,424
              ReLU-6           [-1, 44, 32, 32]               0
       BatchNorm2d-7           [-1, 44, 32, 32]              88
           Dropout-8           [-1, 44, 32, 32]               0
            Conv2d-9           [-1, 44, 16, 16]          17,424
             ReLU-10           [-1, 44, 16, 16]               0
      BatchNorm2d-11           [-1, 44, 16, 16]              88
          Dropout-12           [-1, 44, 16, 16]               0
           Conv2d-13           [-1, 44, 16, 16]          17,424
             ReLU-14           [-1, 44, 16, 16]               0
      BatchNorm2d-15           [-1, 44, 16, 16]              88
          Dropout-16           [-1, 44, 16, 16]               0
           Conv2d-17           [-1, 44, 16, 16]          17,424
             ReLU-18           [-1, 44, 16, 16]               0
      BatchNorm2d-19           [-1, 44, 16, 16]              88
          Dropout-20           [-1, 44, 16, 16]               0
           Conv2d-21             [-1, 44, 8, 8]          17,424
             ReLU-22             [-1, 44, 8, 8]               0
      BatchNorm2d-23             [-1, 44, 8, 8]              88
          Dropout-24             [-1, 44, 8, 8]               0
           Conv2d-25             [-1, 44, 8, 8]          17,424
             ReLU-26             [-1, 44, 8, 8]               0
      BatchNorm2d-27             [-1, 44, 8, 8]              88
          Dropout-28             [-1, 44, 8, 8]               0
           Conv2d-29             [-1, 44, 8, 8]          17,424
             ReLU-30             [-1, 44, 8, 8]               0
      BatchNorm2d-31             [-1, 44, 8, 8]              88
          Dropout-32             [-1, 44, 8, 8]               0
           Conv2d-33             [-1, 44, 4, 4]          17,424
             ReLU-34             [-1, 44, 4, 4]               0
      BatchNorm2d-35             [-1, 44, 4, 4]              88
          Dropout-36             [-1, 44, 4, 4]               0
           Conv2d-37             [-1, 44, 4, 4]          17,424
             ReLU-38             [-1, 44, 4, 4]               0
      BatchNorm2d-39             [-1, 44, 4, 4]              88
          Dropout-40             [-1, 44, 4, 4]               0
           Conv2d-41             [-1, 44, 4, 4]          17,424
             ReLU-42             [-1, 44, 4, 4]               0
      BatchNorm2d-43             [-1, 44, 4, 4]              88
          Dropout-44             [-1, 44, 4, 4]               0
           Conv2d-45             [-1, 44, 4, 4]             440
           Conv2d-46             [-1, 44, 4, 4]           1,980
             ReLU-47             [-1, 44, 4, 4]               0
      BatchNorm2d-48             [-1, 44, 4, 4]              88
          Dropout-49             [-1, 44, 4, 4]               0
        AvgPool2d-50             [-1, 44, 1, 1]               0
           Linear-51                   [-1, 10]             440
================================================================
Total params: 179,344
Trainable params: 179,344
Non-trainable params: 0
----------------------------------------------------------------

```

## Analysis 

* After multiple tries and intuitions, i felt having dilated convs at middle or initial stages would yield good results 
* having dsc at end is better as same 3*3 is used to extract features 
* gap with 6 would have been better but couldnt figure a way to get it without impacting image features. so went with 4. 
* instead of FC i used 1*1 convolution 
* my guess is because of more number of parameters to be optimised, it took more epochs. 
* with 3 strided convs our __RF__ increases __16__ at each conv increasing __RF__ to __88__ 

```
k = 44

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(                  #in_s, out_s, rf_in, rf_out
            nn.Conv2d(3, k, 3, padding=1, bias=False),           #32*32, 32*32, 1,3
            nn.ReLU(),
            nn.BatchNorm2d(k),
            nn.Dropout(0.01),

            nn.Conv2d(k, k, 3, padding=1, bias=False),           #32*32, 32*32, 3,5
            nn.ReLU(),
            nn.BatchNorm2d(k),
            nn.Dropout(0.01),

            # strided convolution with 2
            nn.Conv2d(k, k, 3,stride=2, padding=1, bias=False),           #32*32, 16*16, 5,6
            nn.ReLU(),
            nn.BatchNorm2d(k),
            nn.Dropout(0.01),

            # nn.Conv2d(40, 24, 1, bias=False),

                               #14*14, 14*14, 8,8

        )

        self.conv2 = nn.Sequential(

            # nn.ReLU(), -->why relu. aggregators dont need relu to be appplied
            # nn.BatchNorm2d(16),
            # nn.Dropout(0.1),

            nn.Conv2d(k, k, 3, padding=1, bias=False),           #16*16, 16*16, 6,10
            nn.ReLU(),
            nn.BatchNorm2d(k),
            nn.Dropout(0.01),

            
            #dilated convolution 
            nn.Conv2d(k, k, 3,dilation=2, padding=2, bias=False),   #16*16, 16*16, 10,18 (kernel - 5)
            nn.ReLU(),
            nn.BatchNorm2d(k),
            nn.Dropout(0.01),


            # strided convolution with 2
            nn.Conv2d(k, k, 3,stride=2, padding=1, bias=False),           #16*16, 8*8, 18,20
            nn.ReLU(),
            nn.BatchNorm2d(k),
            nn.Dropout(0.01),

            # nn.Conv2d(32, 32, 1, bias=False),
        )

        self.conv3 = nn.Sequential(

            # nn.ReLU(), -->why relu. aggregators dont need relu to be appplied
            # nn.BatchNorm2d(16),
            # nn.Dropout(0.1),

            nn.Conv2d(k, k, 3, padding=1, bias=False),           #8*8, 8*8, 20,28
            nn.ReLU(),
            nn.BatchNorm2d(k),
            nn.Dropout(0.01),

            nn.Conv2d(k, k, 3, padding=1, bias=False),           #8*8, 8*8, 28,36
            nn.ReLU(),
            nn.BatchNorm2d(k),
            nn.Dropout(0.01),

            # strided convolution with 2
            nn.Conv2d(k, k, 3,stride=2, padding=1, bias=False),           #8*8, 4*4, 36,40
            nn.ReLU(),
            nn.BatchNorm2d(k),
            nn.Dropout(0.01),

            # nn.Conv2d(32, 24, 1, bias=False),
        )

            # nn.ReLU(),

        self.conv4 = nn.Sequential(
            
            nn.Conv2d(k, k, 3, padding=1, bias=False),           #4*4, 4*4, 40,56
            nn.ReLU(),
            nn.BatchNorm2d(k),
            nn.Dropout(0.01),


            # depthwise seperated convolution 
            nn.Conv2d(k, k, kernel_size=3, stride=1, padding=1, groups=k),  #4*4, 4*4, 56,72
            nn.Conv2d(k, k, kernel_size=1),

            nn.ReLU(),
            nn.BatchNorm2d(k),
            nn.Dropout(0.01),

            nn.Conv2d(k, k, 3, padding=1, bias=False),           #4*4, 4*4, 72,88
            nn.ReLU(),
            nn.BatchNorm2d(k),
            nn.Dropout(0.01),

            nn.AvgPool2d(4),

        )

        self.lin = nn.Linear(k, 10, bias=False)


    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        # x = x + self.resnetconv(x)
        # x = self.trans2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # print(x.shape)
        x = self.lin(x.view(x.size(0), k))
        x = x.view(-1, 10)

        # x = x.view(-1, 16*7*7)
        # x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
```

## Graphs and Misclassified images 

* loss graphs

![alt text](https://github.com/srikanthp1/S9/blob/master/images/accuracy_loss_graphs.png)

* misclassified 

![alt text](https://github.com/srikanthp1/S9/blob/master/images/misclassified.png)