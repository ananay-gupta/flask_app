from flask import Flask, jsonify, request, render_template
import flask
import os                      
import numpy as np             
import pandas as pd            
import torch                   
import torch.nn as nn         
from torch.utils.data import DataLoader
from PIL import Image           
import torch.nn.functional as F 
import torchvision.transforms as transforms  
from torchvision.utils import make_grid       
from torchvision.datasets import ImageFolder  
from torchsummary import summary  
import matplotlib.pyplot as plt
import shutil



data_dir = r"model/Dataset/Plant Diseases Dataset"
train_dir = data_dir + "/train"
valid_dir = data_dir + "/valid"
diseases = os.listdir(train_dir)

train = ImageFolder(train_dir, transform=transforms.ToTensor())
valid = ImageFolder(valid_dir, transform=transforms.ToTensor())   

batch_size = 32
train_dl = DataLoader(train, batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_dl = DataLoader(valid, batch_size, num_workers=2, pin_memory=True)

device=torch.device('cpu')



def delete_files_in_directory(directory_path):
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
     

class SimpleResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        return self.relu2(out) + x

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def predict_image(img, model):
    
    xb = to_device(img.unsqueeze(0), torch.device('cpu'))
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return preds.item()

class ImageClassificationBase(nn.Module):

    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  
        loss = F.cross_entropy(out, labels) 
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                  
        loss = F.cross_entropy(out, labels) 
        acc = accuracy(out, labels)         
        return {"val_loss": loss.detach(), "val_accuracy": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()      
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy} 

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy']))

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True) 
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

        self.conv3 = ConvBlock(128, 256, pool=True) 
        self.conv4 = ConvBlock(256, 512, pool=True) 
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(512, num_diseases))

    def forward(self, xb): 
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

model =  ResNet9(3,38)
model.load_state_dict(torch.load(r"model/Saved Model/plant-disease-model.pth", map_location=torch.device('cpu')),strict=False)
model.eval()
model.cpu()




app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '\test'

@app.route('/', methods = ['GET', 'POST'])
def home():
    if(request.method == 'GET'):

        return render_template("predict.html")

@app.route('/predict', methods = ['POST'])
def predict():

    if 'image' not in request.files:
        return "No image uploaded", 400
    
    uploaded_image = request.files['image']

    if uploaded_image.filename == '':
        return "No selected file", 400
    else:
        old_dir = "test/"+uploaded_image.filename
        new_dir = "image_uploaded/upload/"+uploaded_image.filename
        delete_files_in_directory('image_uploaded/upload')
        shutil.copyfile(old_dir,new_dir)

        transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        ])
        test_data = ImageFolder(root='image_uploaded', transform=transform)
        img, label = test_data[0]
        prediction=diseases[predict_image(img, model)]
        print(prediction)

        return jsonify({"res": prediction.capitalize()})
        

if __name__ == '__main__' :
    app.run(debug = True)



