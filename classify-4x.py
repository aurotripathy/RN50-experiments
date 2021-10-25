# mostly hacked from
# https://learnopencv.com/fully-convolutional-image-classification-on-arbitrary-sized-image/?unapproved=4876&moderation-hash=f6e6f3a86080b1393fb2f38c0d8e082c#comment-4876

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torch.hub import load_state_dict_from_url
from pudb import set_trace


class FullyConvolutionalResnet18(models.ResNet):
    def __init__(self, num_classes=1000, pretrained=False, **kwargs):

        # Start with standard resnet18 defined here
        super().__init__(block=models.resnet.BasicBlock, layers=[
            2, 2, 2, 2], num_classes=num_classes, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(
                models.resnet.model_urls["resnet18"], progress=True)
            self.load_state_dict(state_dict)

        # Replace AdaptiveAvgPool2d with standard AvgPool2d
        self.avgpool = nn.AvgPool2d((7, 7))

        # Convert the original fc layer to a convolutional layer.
        self.last_conv = torch.nn.Conv2d(
            in_channels=self.fc.in_features, out_channels=num_classes, kernel_size=1)
        self.last_conv.weight.data.copy_(
            self.fc.weight.data.view(*self.fc.weight.data.shape, 1, 1))
        self.last_conv.bias.data.copy_(self.fc.bias.data)

    # Reimplementing forward pass.
#    def _forward_impl(self, x):
    def forward(self, x):
        # Standard forward for resnet18
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        # Notice, there is no forward pass through the original fully connected layer.
        # Instead, we forward pass through the last conv layer
        x = self.last_conv(x)
        return x


# Read image
filename = 'camel-1024x387.jpg'
original_image = Image.open(filename)
print('Original size W H:', original_image.size)

# Transform input image 1. Convert to Tensor 2. Subtract mean 3. Divide by standard deviation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),  # Convert image to tensor.
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # Subtract mean
        std=[0.229, 0.224, 0.225]     # Divide by standard deviation
    )])

image = transform(original_image)

image1 = image[:, :, 0:220].unsqueeze(0)
image2 = image[:, :, 220:440].unsqueeze(0)
image3 = image[:, :, 440:].unsqueeze(0)


print('Shape of image:', image.shape)

if __name__ == "__main__":

    # Read ImageNet class id to name mapping
    with open('imagenet_classes.txt') as f:
        labels = [line.strip() for line in f.readlines()]

# Load modified resnet18 model with pretrained ImageNet weights
model = FullyConvolutionalResnet18(pretrained=True).eval()

with torch.no_grad():
    # Perform inference.
    # Instead of a 1x1000 vector, we will get a
    # 1x1000xnxm output ( i.e. a probabibility map
    # of size n x m for each 1000 class,
    # where n and m depend on the size of the image.)
    preds1 = model(image1)
    preds2 = model(image2)
    preds3 = model(image3)
    print('Response map shape 1 : ', preds1.shape)
    print('Response map shape 2 : ', preds2.shape)
    print('Response map shape 3 : ', preds3.shape)
    # set_trace()
    preds = torch.cat([preds1, preds2, preds3], dim=2)

    # Find the class with the maximum score in the n x m output map
    pred, class_idx = torch.max(preds, dim=1)
    print(class_idx)

    row_max, row_idx = torch.max(pred, dim=1)
    col_max, col_idx = torch.max(row_max, dim=1)
    predicted_class = class_idx[0, row_idx[0, col_idx], col_idx]

    # Print top predicted class
    print('Predicted Class : ', labels[predicted_class], predicted_class)
