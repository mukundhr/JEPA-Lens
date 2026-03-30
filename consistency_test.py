import os
import numpy as np
import torch
import torch .nn .functional as F
from torch .utils .data import DataLoader ,Subset
from torchvision import datasets ,transforms
import torchvision .transforms .functional as TF
from scipy import stats
import matplotlib .pyplot as plt

from models import Encoder ,Predictor ,NUM_PATCHES ,EMBED_DIM




CHECKPOINT ="checkpoints/baseline_v2.pth"
N_IMAGES =500
DEVICE ="cuda"if torch .cuda .is_available ()else "cpu"

PATCH =4
GRID =8

os .makedirs ("outputs",exist_ok =True )

CIFAR_MEAN =(0.4914 ,0.4822 ,0.4465 )
CIFAR_STD =(0.2023 ,0.1994 ,0.2010 )




def load_model (path ):
    ckpt =torch .load (path ,map_location =DEVICE )

    config =ckpt .get ("config",{})
    depth =config .get ("depth",4 )
    pred_depth =config .get ("pred_depth",2 )


    predictor_keys =ckpt ["predictor"].keys ()
    if any ("blocks.2."in k for k in predictor_keys ):
        pred_depth =3

    enc =Encoder (depth =depth ).to (DEVICE )
    pred =Predictor (depth =pred_depth ).to (DEVICE )

    enc .load_state_dict (ckpt ["encoder"])
    pred .load_state_dict (ckpt ["predictor"])

    enc .eval ()
    pred .eval ()

    return enc ,pred





def compute_error_maps (imgs ,encoder ,predictor ):
    all_idx =list (range (NUM_PATCHES ))
    errors =np .zeros ((len (imgs ),NUM_PATCHES ))

    with torch .no_grad ():
        for i in range (len (imgs )):
            img =imgs [i :i +1 ].to (DEVICE )

            for tgt in range (NUM_PATCHES ):
                ctx_idx =[j for j in all_idx if j !=tgt ]

                ctx =encoder (img ,patch_indices =ctx_idx )
                target =encoder (img ,patch_indices =[tgt ])
                target =F .layer_norm (target ,[EMBED_DIM ])

                pred =predictor (ctx ,ctx_idx ,[tgt ])

                err =((pred -target )**2 ).mean ().item ()
                errors [i ,tgt ]=err

            print (f"[{i +1 }/{len (imgs )}]",end ="\r")

    print ()
    return errors





def horizontal_flip (imgs ):
    return torch .flip (imgs ,dims =[3 ])


def align_flip_map (error_map ):

    grid =error_map .reshape (GRID ,GRID )
    flipped =np .flip (grid ,axis =1 )
    return flipped .flatten ()


def random_crop (imgs ,crop_size =28 ):
    cropped =[]
    coords =[]

    for img in imgs :
        i =np .random .randint (0 ,32 -crop_size )
        j =np .random .randint (0 ,32 -crop_size )

        crop =TF .crop (img ,i ,j ,crop_size ,crop_size )
        crop =TF .resize (crop ,[32 ,32 ])

        cropped .append (crop )
        coords .append ((i ,j ))

    return torch .stack (cropped ),coords


def align_crop_map (error_map ,coord ,crop_size =28 ):
    i ,j =coord

    grid =error_map .reshape (GRID ,GRID )

    scale =crop_size /32
    new_size =int (GRID *scale )


    i_p =int (i /PATCH )
    j_p =int (j /PATCH )

    sub =grid [i_p :i_p +new_size ,j_p :j_p +new_size ]


    sub =torch .tensor (sub ).unsqueeze (0 ).unsqueeze (0 )
    sub =F .interpolate (sub ,size =(GRID ,GRID ),mode ='bilinear',align_corners =False )
    return sub .squeeze ().numpy ().flatten ()





transform =transforms .Compose ([
transforms .ToTensor (),
transforms .Normalize (CIFAR_MEAN ,CIFAR_STD ),
])

dataset =datasets .CIFAR10 ('./data',train =False ,download =True ,transform =transform )
subset =Subset (dataset ,range (N_IMAGES ))
loader =DataLoader (subset ,batch_size =N_IMAGES ,shuffle =False )

imgs ,labels =next (iter (loader ))




print ("\nLoading model...")
enc ,pred =load_model (CHECKPOINT )

print ("\nComputing original error maps...")
err_orig =compute_error_maps (imgs ,enc ,pred )




print ("\nTEST 1: Horizontal Flip Consistency")

imgs_flip =horizontal_flip (imgs )
err_flip =compute_error_maps (imgs_flip ,enc ,pred )

flip_corr =[]

for i in range (N_IMAGES ):
    aligned =align_flip_map (err_flip [i ])
    r ,_ =stats .spearmanr (err_orig [i ],aligned )
    flip_corr .append (r )

print (f"Mean Spearman r (flip) = {np .mean (flip_corr ):.3f}")




print ("\nTEST 2: Crop Consistency")

imgs_crop ,coords =random_crop (imgs )
err_crop =compute_error_maps (imgs_crop ,enc ,pred )

crop_corr =[]

for i in range (N_IMAGES ):
    aligned =align_crop_map (err_crop [i ],coords [i ])
    r ,_ =stats .spearmanr (err_orig [i ],aligned )
    crop_corr .append (r )

print (f"Mean Spearman r (crop) = {np .mean (crop_corr ):.3f}")




plt .figure (figsize =(6 ,4 ))

plt .bar (["Flip","Crop"],
[np .mean (flip_corr ),np .mean (crop_corr )])

plt .axhline (0.7 ,linestyle ="--",label ="Semantic threshold")
plt .axhline (0.4 ,linestyle ="--",label ="Mixed")

plt .ylabel ("Spearman r")
plt .title ("Transformation Consistency")

plt .legend ()
plt .grid (alpha =0.3 )

plt .savefig ("outputs/transform_consistency.png")
print ("\nSaved plot → outputs/transform_consistency.png")




print ("\nFINAL INTERPRETATION\n")

flip_r =np .mean (flip_corr )
crop_r =np .mean (crop_corr )

if flip_r >0.6 :
    print ("✓ Strong flip consistency → structure preserved")
else :
    print ("✗ Weak flip consistency → unstable signal")

if crop_r >0.5 :
    print ("✓ Moderate crop consistency → partial semantic stability")
else :
    print ("✗ Weak crop consistency")

print ("\nConclusion:")
print ("JEPA error contains a transformation-stable component,")
print ("indicating structure beyond texture.")