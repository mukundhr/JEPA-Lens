import argparse ,os
import numpy as np
import torch
import torch .nn as nn
import torch .nn .functional as F
from torch .utils .data import DataLoader ,Subset
from torchvision import datasets ,transforms
import matplotlib .pyplot as plt
import matplotlib .gridspec as gridspec
from sklearn .manifold import TSNE
from sklearn .linear_model import LogisticRegression
from sklearn .preprocessing import StandardScaler
from sklearn .metrics import accuracy_score

from models import Encoder ,Predictor ,NUM_PATCHES ,EMBED_DIM


parser =argparse .ArgumentParser ()
parser .add_argument ("--checkpoint",default ="checkpoints/baseline.pth")
args =parser .parse_args ()

DEVICE ="cuda"if torch .cuda .is_available ()else "cpu"
CIFAR_CLASSES =['plane','car','bird','cat','deer',
'dog','frog','horse','ship','truck']

os .makedirs ("outputs",exist_ok =True )


print (f"Loading checkpoint: {args .checkpoint }")
ckpt =torch .load (args .checkpoint ,map_location =DEVICE ,weights_only =False )
variant =ckpt .get ("config",{}).get ("variant","baseline")
depth =ckpt .get ("config",{}).get ("depth",4 )
pred_depth =ckpt .get ("config",{}).get ("pred_depth",2 )


if "blocks.2."in str (list (ckpt ["predictor"].keys ())):
    pred_depth =3

print (f"Variant    : {variant }")
print (f"Enc depth  : {depth }")
print (f"Pred depth : {pred_depth }\n")

encoder =Encoder (depth =depth ).to (DEVICE )
encoder .load_state_dict (ckpt ["encoder"])
encoder .eval ()


eval_transform =transforms .Compose ([
transforms .ToTensor (),
transforms .Normalize ((0.4914 ,0.4822 ,0.4465 ),
(0.2023 ,0.1994 ,0.2010 )),
])

train_data =datasets .CIFAR10 ('./data',train =True ,download =True ,transform =eval_transform )
test_data =datasets .CIFAR10 ('./data',train =False ,download =True ,transform =eval_transform )


train_loader =DataLoader (train_data ,batch_size =256 ,shuffle =False ,num_workers =0 )
test_loader =DataLoader (Subset (test_data ,range (1000 )),batch_size =256 ,
shuffle =False ,num_workers =0 )



def extract_representations (loader ,encoder ,device ):
    reprs ,labels =[],[]
    with torch .no_grad ():
        for imgs ,lbls in loader :
            imgs =imgs .to (device )
            tokens =encoder (imgs )
            pooled =tokens .mean (dim =1 )
            reprs .append (pooled .cpu ().numpy ())
            labels .append (lbls .numpy ())
    return np .concatenate (reprs ),np .concatenate (labels )

print ("Extracting representations...")
train_repr ,train_labels =extract_representations (train_loader ,encoder ,DEVICE )
test_repr ,test_labels =extract_representations (test_loader ,encoder ,DEVICE )
print (f"  Train: {train_repr .shape } | Test: {test_repr .shape }")





print ("\n── CHECK 1: Linear Probe ─────────────────────────────────────────────")

scaler =StandardScaler ()
train_repr_scaled =scaler .fit_transform (train_repr )
test_repr_scaled =scaler .transform (test_repr )

clf =LogisticRegression (max_iter =1000 ,C =1.0 ,random_state =42 )
clf .fit (train_repr_scaled ,train_labels )

train_acc =accuracy_score (train_labels ,clf .predict (train_repr_scaled ))*100
test_acc =accuracy_score (test_labels ,clf .predict (test_repr_scaled ))*100

print (f"  Train accuracy : {train_acc :.1f}%")
print (f"  Test  accuracy : {test_acc :.1f}%")
print (f"  Random baseline: 10.0%")

if test_acc >40 :
    verdict ="✓ STRONG — encoder learned meaningful semantic structure"
elif test_acc >25 :
    verdict ="~ MODERATE — some structure learned, but representations shallow"
else :
    verdict ="✗ WEAK — encoder may not have learned meaningful features yet"
print (f"  Verdict: {verdict }\n")





print ("── CHECK 2: t-SNE Visualization ──────────────────────────────────────")
print ("  Running t-SNE on 1000 test representations (this takes ~1-2 min)...")

tsne =TSNE (n_components =2 ,random_state =42 ,perplexity =30 ,max_iter =1000 )
emb_2d =tsne .fit_transform (test_repr_scaled )

colors =plt .colormaps ['tab10'].resampled (10 )
fig ,ax =plt .subplots (figsize =(10 ,8 ))
for cls_id in range (10 ):
    mask =test_labels ==cls_id
    ax .scatter (emb_2d [mask ,0 ],emb_2d [mask ,1 ],
    color =colors (cls_id ),label =CIFAR_CLASSES [cls_id ],
    alpha =0.7 ,s =15 ,linewidths =0 )
ax .legend (markerscale =2 ,fontsize =9 ,loc ='upper right')
ax .set_title (f"t-SNE of JEPA Encoder Representations ({variant })\n"
f"Clustering = encoder learned semantic structure | "
f"Linear probe test acc: {test_acc :.1f}%",fontsize =11 )
ax .set_xlabel ("t-SNE dim 1")
ax .set_ylabel ("t-SNE dim 2")
ax .grid (alpha =0.2 )
plt .tight_layout ()
plt .savefig (f"outputs/tsne_{variant }.png",dpi =150 )
plt .close ()
print (f"  ✓ Saved → outputs/tsne_{variant }.png")





print ("\n── CHECK 3: Understanding Maps (sanity check) ────────────────────────")

predictor =Predictor (depth =pred_depth ).to (DEVICE )
predictor .load_state_dict (ckpt ["predictor"])
predictor .eval ()

viz_loader =DataLoader (test_data ,batch_size =8 ,shuffle =True ,num_workers =0 )
imgs ,labels =next (iter (viz_loader ))
imgs =imgs .to (DEVICE )

def per_patch_error_map (img_batch ,encoder ,predictor ,device ):
    B =img_batch .shape [0 ]
    error_maps =np .zeros ((B ,NUM_PATCHES ))
    all_indices =list (range (NUM_PATCHES ))

    with torch .no_grad ():
        for tgt_i in range (NUM_PATCHES ):
            ctx_idx =[j for j in all_indices if j !=tgt_i ]
            tgt_idx =[tgt_i ]

            ctx_repr =encoder (img_batch ,patch_indices =ctx_idx )
            tgt_repr =encoder (img_batch ,patch_indices =tgt_idx )
            tgt_repr =F .layer_norm (tgt_repr ,[EMBED_DIM ])
            pred_repr =predictor (ctx_repr ,ctx_idx ,tgt_idx )

            err =((pred_repr -tgt_repr )**2 ).mean (dim =-1 )
            error_maps [:,tgt_i ]=err .squeeze (-1 ).cpu ().numpy ()

    return error_maps

print ("  Computing per-patch error maps for 8 images (64 forward passes each)...")
error_maps =per_patch_error_map (imgs ,encoder ,predictor ,DEVICE )

G =8
mean =torch .tensor ([0.4914 ,0.4822 ,0.4465 ])
std =torch .tensor ([0.2023 ,0.1994 ,0.2010 ])
imgs_display =(imgs .cpu ()*std .view (3 ,1 ,1 )+mean .view (3 ,1 ,1 )).clamp (0 ,1 )

fig =plt .figure (figsize =(16 ,5 ))
gs =gridspec .GridSpec (2 ,8 ,hspace =0.3 ,wspace =0.1 )

for i in range (8 ):
    ax_img =fig .add_subplot (gs [0 ,i ])
    ax_img .imshow (imgs_display [i ].permute (1 ,2 ,0 ).numpy ())
    ax_img .set_title (CIFAR_CLASSES [labels [i ]],fontsize =8 )
    ax_img .axis ('off')

    ax_map =fig .add_subplot (gs [1 ,i ])
    err_grid =error_maps [i ].reshape (G ,G )
    err_tensor =torch .tensor (err_grid ).unsqueeze (0 ).unsqueeze (0 ).float ()
    err_up =F .interpolate (err_tensor ,size =(32 ,32 ),mode ='bilinear',
    align_corners =False ).squeeze ().numpy ()
    ax_map .imshow (err_up ,cmap ='RdYlGn_r',interpolation ='nearest')
    ax_map .axis ('off')
    if i ==0 :
        ax_map .set_title ('understanding\nmap',fontsize =7 )

fig .suptitle (
f"JEPA-Lens Understanding Maps — {variant }\n"
"Red = high prediction error (complex/uncertain) | "
"Green = low error (predictable)\n"
"Expected: background = green, main object = red",
fontsize =9
)
plt .savefig (f"outputs/understanding_map_{variant }.png",dpi =150 ,bbox_inches ='tight')
plt .close ()
print (f"  ✓ Saved → outputs/understanding_map_{variant }.png")





print ("\n"+"="*55 )
print (f"  EVALUATION SUMMARY — {variant .upper ()}")
print ("="*55 )
print (f"  Linear probe accuracy : {test_acc :.1f}%  (random = 10%)")
print (f"  Verdict               : {verdict }")
print (f"\n  What to check in the output files:")
print (f"  outputs/loss_curve.png")
print (f"  outputs/tsne_{variant }.png")
print (f"  outputs/understanding_map_{variant }.png")
print ("="*55 )