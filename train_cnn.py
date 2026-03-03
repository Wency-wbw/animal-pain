
import argparse, os, torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from data import ImageDataset
from utils import ensure_dir, macro_pr, save_json

def train_epoch(m,loader,crit,opt,dev):
    m.train(); s=0
    for x,y,_ in loader:
        x,y=x.to(dev),y.to(dev); opt.zero_grad(); o=m(x); l=crit(o,y); l.backward(); opt.step(); s+=l.item()*x.size(0)
    return s/len(loader.dataset)

@torch.no_grad()
def eval_epoch(m,loader,dev):
    m.eval(); yt=[]; yp=[]
    for x,y,_ in loader:
        x=x.to(dev); o=m(x).argmax(1).cpu().tolist(); yp+=o; yt+=y.tolist()
    return macro_pr(yt,yp)

def main(a):
    dev="cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir(a.out)
    tr=ImageDataset(a.metadata, a.images_root, split="train", image_size=a.size)
    va=ImageDataset(a.metadata, a.images_root, split="val", image_size=a.size)
    trl=DataLoader(tr,batch_size=a.bs,shuffle=True); val=DataLoader(va,batch_size=a.bs)
    m=models.resnet18(weights=models.ResNet18_Weights.DEFAULT); m.fc=nn.Linear(m.fc.in_features,2); m.to(dev)
    opt=torch.optim.AdamW(m.parameters(), lr=a.lr); crit=nn.CrossEntropyLoss()
    best=0; bestp=os.path.join(a.out,"best.pt")
    for e in range(a.epochs):
        loss=train_epoch(m,trl,crit,opt,dev); met=eval_epoch(m,val,dev)
        if met["accuracy"]>best: best=met["accuracy"]; torch.save(m.state_dict(), bestp)
        save_json(met, os.path.join(a.out, f"metrics_epoch_{e}.json")); print(e, met)
    print("Best acc:", best)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--metadata",required=True); ap.add_argument("--images_root",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--epochs",type=int,default=10); ap.add_argument("--bs",type=int,default=16); ap.add_argument("--lr",type=float,default=1e-3); ap.add_argument("--size",type=int,default=224)
    a=ap.parse_args(); main(a)
