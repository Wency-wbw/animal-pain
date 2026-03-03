
import argparse, os, torch, numpy as np
from torchvision import models, transforms
from PIL import Image, ImageOps
import pandas as pd
from utils import ensure_dir

def build_model(weights):
    m=models.resnet18(weights=models.ResNet18_Weights.DEFAULT); m.fc=torch.nn.Linear(m.fc.in_features,2)
    m.load_state_dict(torch.load(weights, map_location="cpu")); m.eval(); return m

def grad_cam(m, x, target_layer):
    acts={}; grads={}
    def fh(module, inp, out): acts["v"]=out.detach()
    def bh(module, gin, gout): grads["v"]=gout[0].detach()
    h1=target_layer.register_forward_hook(fh); h2=target_layer.register_backward_hook(bh)
    out=m(x); idx=out.argmax(1).item(); score=out[0,idx]; m.zero_grad(); score.backward()
    a=acts["v"][0]; g=grads["v"][0]; w=g.mean(dim=(1,2), keepdim=True); cam=(a*w).sum(0).relu(); cam=(cam-cam.min())/(cam.max()-cam.min()+1e-6)
    h1.remove(); h2.remove(); return cam.numpy()

def overlay(img, cam):
    cam_img=Image.fromarray((cam*255).astype("uint8")).resize(img.size)
    cam_img=ImageOps.colorize(cam_img.convert("L"), black="black", white="white")
    return Image.blend(img.convert("RGB"), cam_img, 0.45)

def main(a):
    os.makedirs(ensure_dir(a.out), exist_ok=True)
    m=build_model(a.model); layer=list(m.children())[-2]
    df=pd.read_csv(a.metadata); ids=set(a.ids)
    t=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    for _,r in df.iterrows():
        if r["id"] not in ids: continue
        img=Image.open(os.path.join(a.images_root, r["path"])).convert("RGB")
        x=t(img).unsqueeze(0); cam=grad_cam(m, x, layer); overlay(img, cam).save(os.path.join(a.out, f"{r['id']}_cam.png"))
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--metadata",required=True); ap.add_argument("--images_root",required=True); ap.add_argument("--model",required=True); ap.add_argument("--ids",nargs="+",required=True); ap.add_argument("--out",required=True)
    a=ap.parse_args(); main(a)
