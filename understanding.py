import argparse ,os
import numpy as np
import torch
import torch .nn .functional as F
from torch .utils .data import DataLoader
from torchvision import datasets ,transforms
import matplotlib .pyplot as plt
import matplotlib .gridspec as gridspec

from models import Encoder ,Predictor ,NUM_PATCHES ,EMBED_DIM


parser =argparse .ArgumentParser ()
parser .add_argument ("--checkpoint",default ="checkpoints/baseline.pth")
parser .add_argument ("--n_images",default =8 ,type =int )
args =parser .parse_args ()

DEVICE ="cuda"if torch .cuda .is_available ()else "cpu"
CIFAR_CLASSES =['plane','car','bird','cat','deer',
'dog','frog','horse','ship','truck']
GRID =8
os .makedirs ("outputs",exist_ok =True )


ckpt =torch .load (args .checkpoint ,map_location =DEVICE )
variant =ckpt .get ("config",{}).get ("variant","baseline")

encoder =Encoder (depth =4 ).to (DEVICE )
predictor =Predictor (depth =2 ).to (DEVICE )
encoder .load_state_dict (ckpt ["encoder"])
predictor .load_state_dict (ckpt ["predictor"])
encoder .eval ()
predictor .eval ()


transform =transforms .Compose ([
transforms .ToTensor (),
transforms .Normalize ((0.4914 ,0.4822 ,0.4465 ),
(0.2023 ,0.1994 ,0.2010 )),
])
test_data =datasets .CIFAR10 ('./data',train =False ,download =True ,transform =transform )
loader =DataLoader (test_data ,batch_size =args .n_images ,shuffle =True )
imgs ,labels =next (iter (loader ))
imgs =imgs .to (DEVICE )





def per_patch_map (img_batch ,encoder ,predictor ):
    B =img_batch .shape [0 ]
    errors =np .zeros ((B ,NUM_PATCHES ))
    all_idx =list (range (NUM_PATCHES ))

    with torch .no_grad ():
        for tgt_i in range (NUM_PATCHES ):
            ctx_idx =[j for j in all_idx if j !=tgt_i ]
            tgt_idx =[tgt_i ]

            ctx =encoder (img_batch ,patch_indices =ctx_idx )
            tgt =encoder (img_batch ,patch_indices =tgt_idx )
            tgt =F .layer_norm (tgt ,[EMBED_DIM ])
            pred =predictor (ctx ,ctx_idx ,tgt_idx )

            err =((pred -tgt )**2 ).mean (dim =-1 ).squeeze (-1 )
            errors [:,tgt_i ]=err .cpu ().numpy ()

    return errors





def sliding_window_map (img_batch ,encoder ,predictor ,window =2 ,stride =1 ):
    B =img_batch .shape [0 ]
    n_pos =(GRID -window )//stride +1
    errors =np .zeros ((B ,n_pos ,n_pos ))

    with torch .no_grad ():
        for row in range (n_pos ):
            for col in range (n_pos ):

                tgt_idx =[]
                for dr in range (window ):
                    for dc in range (window ):
                        patch_id =(row *stride +dr )*GRID +(col *stride +dc )
                        tgt_idx .append (patch_id )
                tgt_idx =sorted (tgt_idx )
                ctx_idx =sorted ([j for j in range (NUM_PATCHES )if j not in tgt_idx ])

                ctx =encoder (img_batch ,patch_indices =ctx_idx )
                tgt =encoder (img_batch ,patch_indices =tgt_idx )
                tgt =F .layer_norm (tgt ,[EMBED_DIM ])
                pred =predictor (ctx ,ctx_idx ,tgt_idx )

                err =((pred -tgt )**2 ).mean (dim =(1 ,2 ))
                errors [:,row ,col ]=err .cpu ().numpy ()

    return errors





print (f"Generating per-patch maps ({args .n_images } images × 64 passes)...")
pp_errors =per_patch_map (imgs ,encoder ,predictor )

print (f"Generating sliding window maps ({args .n_images } images × 49 passes)...")
sw_errors =sliding_window_map (imgs ,encoder ,predictor )





mean =torch .tensor ([0.4914 ,0.4822 ,0.4465 ])
std =torch .tensor ([0.2023 ,0.1994 ,0.2010 ])
imgs_display =(imgs .cpu ()*std .view (3 ,1 ,1 )+mean .view (3 ,1 ,1 )).clamp (0 ,1 )

N =args .n_images
fig =plt .figure (figsize =(N *2.2 ,7 ))
gs =gridspec .GridSpec (3 ,N ,hspace =0.35 ,wspace =0.05 )


pp_min ,pp_max =pp_errors .min (),pp_errors .max ()
sw_min ,sw_max =sw_errors .min (),sw_errors .max ()

for i in range (N ):

    ax =fig .add_subplot (gs [0 ,i ])
    ax .imshow (imgs_display [i ].permute (1 ,2 ,0 ).numpy ())
    ax .set_title (CIFAR_CLASSES [labels [i ]],fontsize =8 ,pad =2 )
    ax .axis ('off')
    if i ==0 :
        ax .set_ylabel ("Original",fontsize =8 ,rotation =0 ,labelpad =40 ,va ='center')


    ax =fig .add_subplot (gs [1 ,i ])
    pp_grid =pp_errors [i ].reshape (GRID ,GRID )
    pp_norm =(pp_grid -pp_min )/(pp_max -pp_min +1e-8 )
    pp_tensor =torch .tensor (pp_norm ).unsqueeze (0 ).unsqueeze (0 ).float ()
    pp_up =F .interpolate (pp_tensor ,size =(32 ,32 ),
    mode ='nearest').squeeze ().numpy ()
    ax .imshow (pp_up ,cmap ='RdYlGn_r',vmin =0 ,vmax =1 )
    ax .axis ('off')
    if i ==0 :
        ax .set_ylabel ("Per-patch\n(blocky)",fontsize =8 ,rotation =0 ,
        labelpad =40 ,va ='center')


    ax =fig .add_subplot (gs [2 ,i ])
    sw_norm =(sw_errors [i ]-sw_min )/(sw_max -sw_min +1e-8 )
    sw_tensor =torch .tensor (sw_norm ).unsqueeze (0 ).unsqueeze (0 ).float ()
    sw_up =F .interpolate (sw_tensor ,size =(32 ,32 ),
    mode ='bilinear',align_corners =False ).squeeze ().numpy ()
    ax .imshow (sw_up ,cmap ='RdYlGn_r',vmin =0 ,vmax =1 )
    ax .axis ('off')
    if i ==0 :
        ax .set_ylabel ("Sliding\nwindow",fontsize =8 ,rotation =0 ,
        labelpad =40 ,va ='center')

fig .suptitle (
f"JEPA-Lens: Understanding Maps — {variant .upper ()}\n"
"Red = high prediction error (complex, uncertain)   |   "
"Green = low error (predictable, trivial)\n"
"Look for: object regions red, background regions green",
fontsize =10
)
out_path =f"outputs/understanding_maps_comparison_{variant }.png"
plt .savefig (out_path ,dpi =150 ,bbox_inches ='tight')
plt .close ()
print (f"\n✓ Saved → {out_path }")
print ("\nWhat to look for:")
print ("  Per-patch and sliding window should roughly agree on which regions are complex.")
print ("  Where they agree = robust signal.")
print ("  Where they disagree = likely a boundary artifact from patch size.")