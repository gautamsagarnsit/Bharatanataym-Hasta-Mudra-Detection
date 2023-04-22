import numpy as np
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix,recall_score


device="cuda"
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

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
    def __init__(self,csv_file,root_dir,transform=None):
        self.annotations=pd.read_csv(csv_file) #Labels Dir
        self.root_dir=root_dir # Image Dir
        self.transform=transform
        self.nc=self.annotations['Label'].max()+1
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,self.annotations.iloc[idx, 0])
        image = io.imread(img_name)

        lab = self.annotations.iloc[idx, 1:]
        label=np.zeros((self.nc,1))
        label[int(lab)]=1
        label = np.array(label).astype('float')
        #if self.transform:
            #sample['image'] = self.transform(sample['image'])
        image=torchvision.transforms.functional.to_tensor(image)
        #image=image.permute(2, 0, 1)
        return image,label

trainset = MudraDataset(root_dir= "Data/bht_mudra",csv_file="Data/train.csv",transform=transform_train)
testset = MudraDataset(root_dir= "Data/bht_mudra",csv_file="Data/test.csv",transform=transform_test)
print("Length of TrainSet:",len(trainset))
print("Length of ValSet:",len(testset))

trainloader=DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
valloader=DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

config = dict(
    epochs=200,
    nc=len(classes),
    batch_size=32,
    learning_rate=0.0001,
    dataset="Mudra",
    architecture="ResNet18")

net=models.resnet18(pretrained=False) 
net.fc=nn.Linear(512,config['nc'],bias=True)
momentum=0.9
loss=nn.CrossEntropyLoss()
#optimizer=optim.SGD(net.parameters(),lr=config['learning_rate'],momentum=momentum) 
optimizer=optim.Adam(net.parameters(),lr=config['learning_rate'])
scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=int(config['epochs']/3),gamma=10)


if not os.path.exists('logs_resnet'):
    os.mkdir('logs_resnet')

net = net.to(device)

if os.path.exists('logs_resnet/best_ckpt.pth'):
    #Add code
    checkpoint = torch.load("logs_resnet/best_ckpt.pth")
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
elif os.path.exists('logs_resnet/last_ckpt.pth'):
    checkpoint = torch.load("logs_resnet/last_ckpt.pth")
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


#### net = Invoke pretrained ResNet18 model #######
# Training
def train(epoch):   
    print('Training Epoch: %d' % epoch)
    net.train()
    step=0
    i=0
    true_prediction=0
    total_sample_size=0
    loss_sum=0
    #for batch_idx, (inputs, targets) in enumerate(trainloader):
    with tqdm(trainloader, unit="batch") as tepoch:
        for inputs, targets in tepoch:
        #print(batch_idx)
          inputs = inputs.to(device)
          #targets=torch.zeros((inputs.shape[0],config['nc']))
          #for idx,row in enumerate(targets):
          #  row[tar[idx]]=1
          targets=targets.to(device)
          targets=targets.squeeze()
          # Write your code here
          #print(inputs.shape)
          out = net(inputs)
          step+=len(inputs)
          #out=out.reshape(-1,-1,1)
          #print(out.shape,targets.shape)
          #print(out,targets)
          l=loss(out,targets)
          loss_sum+=l
          i+=1
          prob,prediction=torch.max(out.data,dim=1)
          true_prediction+=torch.sum(torch.stack([prediction==targets.argmax(dim=1)])).item()
          total_sample_size+=targets.size(0)
          optimizer.zero_grad()
          l.backward()
          optimizer.step()
    accuracy=true_prediction/total_sample_size
    avg_loss=loss_sum/i
    print("Training accuracy:",accuracy)
    print("Training Loss:",avg_loss)
    return accuracy,avg_loss.item()

def test(epoch):
    global best_acc
    net.eval()
    true_prediction=0
    total_sample_size=0
    loss_sum=0
    i=0
    all_targets=[]
    all_prediction=[]
    print("Testing Epoch: ",epoch)
    with torch.no_grad():
      with tqdm(valloader, unit="batch") as tepoch:
        for inputs, targets in tepoch:
        #for batch_idx, (inputs, targets) in enumerate(valloader):  
            #print(batch_idx)          
            inputs, targets = inputs.to(device), targets.to(device)
            targets=targets.squeeze()
            all_targets+=[x for x in targets]
            #Code
            out=net(inputs)
            l=loss(out,targets)
            loss_sum+=l
            i+=1
            prob,prediction=torch.max(out.data,dim=1)
            all_prediction+=[x for x in prediction]
            true_prediction+=torch.sum(torch.stack([prediction==targets.argmax(dim=1)])).item()
            total_sample_size+=targets.size(0)
        accuracy=true_prediction/total_sample_size
        avg_loss=loss_sum/i
        print("Validation Accuracy:",accuracy)
        print("Validation Loss:",avg_loss)
        #recall=recall_score(torch.tensor(all_targets).to('cpu'), torch.tensor(all_prediction).to('cpu'), average=None)
        # Save checkpoint for the model which yields best accuracy
        if accuracy>best_acc:
            print("Saving checkpoint with accuracy = ",accuracy*100)
            best_acc=accuracy
            torch.save({
                'epoch':epoch,
                'model_state_dict':net.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss':avg_loss,
                'accuracy':best_acc
                #'recall':recall
            },'logs_resnet/best_ckpt.pth')
        torch.save({
                'epoch':epoch,
                'model_state_dict':net.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss':avg_loss,
                'accuracy':best_acc
                #'recall':recall
            },'logs_resnet/last_ckpt.pth')
        return accuracy,avg_loss.item(),all_targets,all_prediction

def train_test():
    global best_acc
    best_acc=0
    best_score=None
    training_acc=[]
    training_loss=[]
    val_acc=[]
    val_loss=[]
    counter=0
    patience=200
    stop=False
    x=0
    for epoch in range(1,config['epochs']+1):
        #scheduler.step()
    #for epoch in range(1,3):
        print("Training")
        train_acc,train_loss=train(epoch)
        training_acc.append(train_acc)
        training_loss.append(train_loss)
        print("Testing")
        va,vl,at,ap=test(epoch)
        val_acc.append(va)
        val_loss.append(vl)
        x+=1
        if best_score is None:
            best_score=va
        elif va-best_score<0.05:
            counter+=1
            if counter>=patience:
                stop = True
        else:
            best_score=va
            counter=0
        if stop:
            print("Early Stopping")
            break
    return training_acc,training_loss,val_acc,val_loss,at,ap,x

training_acc,training_loss,val_acc,val_loss,targets,predictions,x=train_test()