import pandas as pd
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import time

device='cuda' if torch.cuda.is_available() else 'cpu'

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
classes = ['Alapadmam', 'Katakamukha', 'Kapotham', 'Ardhapathaka', 'Pushpaputa', 'Shukatundam', 'Trishulam', 
           'Nagabandha', 'Padmakosha', 'Sarpasirsha', 'Shanka', 'Kartariswastika', 'Pasha', 'Suchi', 'Varaha',
           'Ardhachandran', 'Kurma', 'Bramaram', 'Hamsapaksha', 'Mayura', 'Mushti', 'Swastika', 'Matsya',
           'Kilaka', 'Katrimukha', 'Hamsasyam', 'Karkatta', 'Khatva', 'Mrigasirsha', 'Katakavardhana', 'Kapith',
           'Chandrakala', 'Mukulam', 'Kangulam', 'Aralam', 'Samputa', 'Tamarachudam', 'Garuda', 'Tripathaka',
           'Chakra', 'Shivalinga', 'Pathaka', 'Sikharam', 'Berunda', 'Simhamukham', 'Chaturam', 'Anjali', 'Sakata']
print("Number of classes:",len(classes))

class MudraDataset(Dataset):
    def __init__(self,root_dir,transform=None):
        self.root_dir=root_dir # Image Dir
        self.transform=transform
        self.nfiles=0
        self.names=[]
        for path in os.listdir(self.root_dir):
            if os.path.isfile(os.path.join(self.root_dir,path)):
                self.names.append(path)
                self.nfiles+=1
    def __len__(self):
        return self.nfiles
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,self.names[idx])
        image = io.imread(img_name)
        #if self.transform:
            #sample['image'] = self.transform(sample['image'])
        image=torchvision.transforms.functional.to_tensor(image)
        #image=image.permute(2, 0, 1)
        return image,self.names[idx]

testset = MudraDataset(root_dir= "Data/test_set",transform=transform_test)

print("Length of ValSet:",len(testset))

testloader=DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

net=models.resnet18(pretrained=False) 
net.fc=nn.Linear(512,len(classes),bias=True)

if not os.path.exists('results_resnet'):
    os.mkdir('results_resnet')

net = net.to(device)

if os.path.exists('logs_resnet/best_ckpt.pth'):
    checkpoint = torch.load("logs_resnet/best_ckpt.pth",map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['model_state_dict'])
elif os.path.exists('logs_resnet/last_ckpt.pth'):
    checkpoint = torch.load("logs_resnet/last_ckpt.pth",map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['model_state_dict'])
else:
    print("Weights Not Found")

if os.path.exists('results_resnet/result.txt'):
    i=1
    while os.path.exists('results_resnet/result'+str(i)+'.txt'):
        i+=1
    res = open("results_resnet/result"+str(i)+".txt","w")
else:
    res = open("results_resnet/result.txt","w")

res.write("filename,predicted label,predicted label idx\n")

net.eval()
den = 0 
num = 0
with torch.no_grad():
    with tqdm(testloader, unit="batch") as tepoch:
        for inputs, filename in tepoch:
            inputs = inputs.to(device)
            start=time.time()
            out=net(inputs)
            prob,prediction=torch.max(out.data,dim=1)
            end = time.time()
            num = end-start
            den+=1
            #for idx,name in enumerate(filename):
            #    print(name,classes[prediction[idx]],str(prediction[idx]))
            #    res.write(name+","+classes[prediction[idx]]+","+str(prediction[idx])+'\n')
res.close()
            
print("Average time per image:",num/den)


