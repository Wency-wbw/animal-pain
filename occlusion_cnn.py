
import argparse, os, torch, numpy as np, pandas as pd
from PIL import Image, ImageDraw
from torchvision import models, transforms
from utils import ensure_dir, save_json

def load_resnet18(weights): 
    m=models.resnet18(weights=models.ResNet18_Weights.DEFAULT); m.fc=torch.nn.Linear(m.fc.in_features,2); m.load_state_dict(torch.load(weights, map_location="cpu")); m.eval(); return m

def mask_rect(img, box):
    im=img.copy(); ImageDraw.Draw(im).rectangle(box, fill=(127,127,127)); return im

def main(a):
    os.makedirs(ensure_dir(a.out), exist_ok=True)
    m=load_resnet18(a.model); dev="cuda" if torch.cuda.is_available() else "cpu"; m.to(dev)
    df=pd.read_csv(a.metadata)
    t=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    def pred(im):
        with torch.no_grad():
            return m(t(im).unsqueeze(0).to(dev)).argmax(1).item()
    yt=[]; y_full=[]; y_hide={"ears":[], "eyes":[], "mouth":[]}; y_mouthonly=[]
    for _,r in df.iterrows():
        p=os.path.join(a.images_root, r["path"]); 
        if not os.path.exists(p): continue
        img=Image.open(p).convert("RGB"); yt.append(int(r["label"])); y_full.append(pred(img))
        W,H=img.size; thirds={(0,"ears"):(0,0,W,H//3),(1,"eyes"):(0,H//3,W,2*H//3),(2,"mouth"):(0,2*H//3,W,H)}
        for _,k in [(0,"ears"),(1,"eyes"),(2,"mouth")]:
            y_hide[k].append(pred(mask_rect(img, thirds[_[0]+(k,)][0])) if False else pred(mask_rect(img, thirds[(0,k)][0])))
        # proper rects
        ears=(0,0,W,H//3); eyes=(0,H//3,W,2*H//3); mouth=(0,2*H//3,W,H)
        y_hide["ears"][-1]=pred(mask_rect(img, ears)); y_hide["eyes"][-1]=pred(mask_rect(img, eyes)); y_hide["mouth"][-1]=pred(mask_rect(img, mouth))
        mimg=Image.new("RGB", img.size, (127,127,127)); mimg.paste(img.crop(mouth), mouth); y_mouthonly.append(pred(mimg))
    from sklearn.metrics import accuracy_score
    res={"full_face":accuracy_score(yt,y_full), "hide_ears":accuracy_score(yt,y_hide["ears"]), "hide_eyes":accuracy_score(yt,y_hide["eyes"]), "hide_mouth":accuracy_score(yt,y_hide["mouth"]), "mouth_only":accuracy_score(yt,y_mouthonly)}
    save_json(res, os.path.join(a.out,"occlusion_cnn.json")); print(res)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--metadata",required=True); ap.add_argument("--images_root",required=True); ap.add_argument("--model",required=True); ap.add_argument("--out",required=True)
    a=ap.parse_args(); main(a)
