import argparse ,os
import numpy as np
import torch
import torch .nn .functional as F
from torch .utils .data import DataLoader ,Subset
from torchvision import datasets ,transforms
import matplotlib .pyplot as plt
from sklearn .linear_model import LogisticRegression
from sklearn .preprocessing import StandardScaler
from sklearn .metrics import accuracy_score

from models import Encoder ,Predictor ,NUM_PATCHES ,EMBED_DIM


parser =argparse .ArgumentParser ()
parser .add_argument ("--checkpoint",default ="checkpoints/baseline_v2.pth")
parser .add_argument ("--k",nargs ="+",type =int ,default =[4 ,8 ,16 ,24 ],
help ="Number of patches to ablate (out of 64)")
parser .add_argument ("--n_images",type =int ,default =2000 ,
help ="Images to use for error map generation (more = slower but more reliable)")
args =parser .parse_args ()

DEVICE ="cuda"if torch .cuda .is_available ()else "cpu"
CIFAR_CLASSES =['plane','car','bird','cat','deer',
'dog','frog','horse','ship','truck']
os .makedirs ("outputs",exist_ok =True )

print (f"Device     : {DEVICE }")
print (f"Checkpoint : {args .checkpoint }")
print (f"k values   : {args .k }")
print (f"N images   : {args .n_images }\n")


ckpt =torch .load (args .checkpoint ,map_location =DEVICE ,weights_only =False )
variant =ckpt .get ("config",{}).get ("variant","baseline")
depth =ckpt .get ("config",{}).get ("depth",4 )
pred_depth =ckpt .get ("config",{}).get ("pred_depth",2 )
if "blocks.2."in str (list (ckpt ["predictor"].keys ())):
    pred_depth =3

encoder =Encoder (depth =depth ).to (DEVICE )
predictor =Predictor (depth =pred_depth ).to (DEVICE )
encoder .load_state_dict (ckpt ["encoder"])
predictor .load_state_dict (ckpt ["predictor"])
encoder .eval ()
predictor .eval ()


transform =transforms .Compose ([
transforms .ToTensor (),
transforms .Normalize ((0.4914 ,0.4822 ,0.4465 ),
(0.2023 ,0.1994 ,0.2010 )),
])
train_data =datasets .CIFAR10 ('./data',train =True ,download =True ,transform =transform )
test_data =datasets .CIFAR10 ('./data',train =False ,download =True ,transform =transform )


probe_train =Subset (train_data ,range (args .n_images ))
probe_test =Subset (test_data ,range (2000 ))

probe_train_loader =DataLoader (probe_train ,batch_size =128 ,shuffle =False ,num_workers =0 )
probe_test_loader =DataLoader (probe_test ,batch_size =128 ,shuffle =False ,num_workers =0 )





def compute_error_maps (loader ,encoder ,predictor ,device ,desc =""):
    all_imgs ,all_errors ,all_labels =[],[],[]
    all_indices =list (range (NUM_PATCHES ))
    total =len (loader )

    with torch .no_grad ():
        for batch_i ,(imgs ,lbls )in enumerate (loader ):
            imgs =imgs .to (device )
            B =imgs .shape [0 ]
            err_batch =np .zeros ((B ,NUM_PATCHES ))

            for tgt_i in range (NUM_PATCHES ):
                ctx_idx =[j for j in all_indices if j !=tgt_i ]
                tgt_idx =[tgt_i ]

                ctx =encoder (imgs ,patch_indices =ctx_idx )
                tgt =encoder (imgs ,patch_indices =tgt_idx )
                tgt =F .layer_norm (tgt ,[EMBED_DIM ])
                pred =predictor (ctx ,ctx_idx ,tgt_idx )

                err =((pred -tgt )**2 ).mean (dim =-1 ).squeeze (-1 )
                err_batch [:,tgt_i ]=err .cpu ().numpy ()

            all_imgs .append (imgs .cpu ())
            all_errors .append (err_batch )
            all_labels .append (lbls .numpy ())

            if (batch_i +1 )%2 ==0 or batch_i ==total -1 :
                print (f"  {desc } batch [{batch_i +1 }/{total }]",end ="\r")

    print ()
    return (torch .cat (all_imgs ),
    np .concatenate (all_errors ),
    np .concatenate (all_labels ))


print ("="*60 )
print ("STEP 1: Computing per-patch error maps")
print (f"  {args .n_images } train images × 64 patches = {args .n_images *64 } forward passes")
print ("="*60 )
train_imgs ,train_errors ,train_labels =compute_error_maps (
probe_train_loader ,encoder ,predictor ,DEVICE ,desc ="train")

print ("\nTest images:")
test_imgs ,test_errors ,test_labels =compute_error_maps (
probe_test_loader ,encoder ,predictor ,DEVICE ,desc ="test")

print (f"\n  Train error maps: {train_errors .shape }")
print (f"  Test  error maps: {test_errors .shape }")





def get_ablated_representations (imgs_tensor ,errors ,encoder ,device ,k ,mode ):
    reprs =[]
    N =imgs_tensor .shape [0 ]


    BSIZE =64
    for start in range (0 ,N ,BSIZE ):
        end =min (start +BSIZE ,N )
        imgs =imgs_tensor [start :end ].to (device )
        errs =errors [start :end ]
        B =imgs .shape [0 ]

        batch_reprs =[]
        with torch .no_grad ():
            for i in range (B ):
                err_i =errs [i ]

                if mode =='high':
                    ranked =np .argsort (err_i )[::-1 ]
                    remove =set (ranked [:k ].tolist ())
                elif mode =='low':
                    ranked =np .argsort (err_i )
                    remove =set (ranked [:k ].tolist ())
                else :
                    remove =set (np .random .choice (NUM_PATCHES ,k ,replace =False ).tolist ())

                keep_idx =sorted ([j for j in range (NUM_PATCHES )if j not in remove ])


                img_batch =imgs [i ].unsqueeze (0 )
                tokens =encoder (img_batch ,patch_indices =keep_idx )
                pooled =tokens .mean (dim =1 ).squeeze (0 )
                batch_reprs .append (pooled .cpu ().numpy ())

        reprs .extend (batch_reprs )
        print (f"  Ablation ({mode }, k={k }): {end }/{N }",end ="\r")

    print ()
    return np .stack (reprs )





def run_probe (train_repr ,train_labels ,test_repr ,test_labels ):
    scaler =StandardScaler ()
    tr_scaled =scaler .fit_transform (train_repr )
    te_scaled =scaler .transform (test_repr )
    clf =LogisticRegression (max_iter =500 ,C =1.0 ,random_state =42 )
    clf .fit (tr_scaled ,train_labels )
    return accuracy_score (test_labels ,clf .predict (te_scaled ))*100


print ("\n"+"="*60 )
print ("STEP 2: Baseline accuracy (no ablation)")
print ("="*60 )


def get_full_repr (imgs_tensor ,encoder ,device ):
    reprs =[]
    for start in range (0 ,len (imgs_tensor ),128 ):
        end =min (start +128 ,len (imgs_tensor ))
        imgs =imgs_tensor [start :end ].to (device )
        with torch .no_grad ():
            tokens =encoder (imgs )
            pooled =tokens .mean (dim =1 )
        reprs .append (pooled .cpu ().numpy ())
    return np .concatenate (reprs )

train_full =get_full_repr (train_imgs ,encoder ,DEVICE )
test_full =get_full_repr (test_imgs ,encoder ,DEVICE )
baseline_acc =run_probe (train_full ,train_labels ,test_full ,test_labels )
print (f"  Baseline accuracy (0 patches ablated): {baseline_acc :.1f}%")





print ("\n"+"="*60 )
print ("STEP 3: Ablation experiment")
print ("  Hypothesis: ablating HIGH-error patches hurts more than LOW or RANDOM")
print ("="*60 )

results ={"k":args .k ,"high":[],"low":[],"random":[]}

for k in args .k :
    print (f"\n── k={k } patches ablated ({k /NUM_PATCHES *100 :.0f}% of image) ──")

    for mode in ['high','low','random']:
        tr =get_ablated_representations (train_imgs ,train_errors ,encoder ,DEVICE ,k ,mode )
        te =get_ablated_representations (test_imgs ,test_errors ,encoder ,DEVICE ,k ,mode )
        acc =run_probe (tr ,train_labels ,te ,test_labels )
        results [mode ].append (acc )
        print (f"  Ablate {mode :6s}: {acc :.1f}%  (drop: {baseline_acc -acc :.1f}%)")





print ("\n"+"="*60 )
print ("STEP 4: Generating results plot")
print ("="*60 )

fig ,axes =plt .subplots (1 ,2 ,figsize =(14 ,5 ))


ax =axes [0 ]
ax .axhline (baseline_acc ,color ='gray',linestyle ='--',linewidth =1.5 ,
label =f'No ablation ({baseline_acc :.1f}%)',alpha =0.7 )
ax .plot (args .k ,results ['high'],'o-',color ='#ef5350',linewidth =2 ,
markersize =7 ,label ='Ablate HIGH-error patches')
ax .plot (args .k ,results ['low'],'s-',color ='#66bb6a',linewidth =2 ,
markersize =7 ,label ='Ablate LOW-error patches')
ax .plot (args .k ,results ['random'],'^-',color ='#7986cb',linewidth =2 ,
markersize =7 ,label ='Ablate RANDOM patches')

ax .set_xlabel ("k (patches ablated)",fontsize =11 )
ax .set_ylabel ("Linear probe accuracy (%)",fontsize =11 )
ax .set_title ("Causal Test: Does prediction error\ntrack semantic importance?",fontsize =12 )
ax .legend (fontsize =9 )
ax .grid (alpha =0.3 )
ax .set_xticks (args .k )


ax =axes [1 ]
drop_high =[baseline_acc -a for a in results ['high']]
drop_low =[baseline_acc -a for a in results ['low']]
drop_random =[baseline_acc -a for a in results ['random']]

ax .plot (args .k ,drop_high ,'o-',color ='#ef5350',linewidth =2 ,
markersize =7 ,label ='Ablate HIGH-error (semantic)')
ax .plot (args .k ,drop_low ,'s-',color ='#66bb6a',linewidth =2 ,
markersize =7 ,label ='Ablate LOW-error (trivial)')
ax .plot (args .k ,drop_random ,'^-',color ='#7986cb',linewidth =2 ,
markersize =7 ,label ='Ablate RANDOM')
ax .axhline (0 ,color ='gray',linestyle ='--',linewidth =1 ,alpha =0.5 )

ax .set_xlabel ("k (patches ablated)",fontsize =11 )
ax .set_ylabel ("Accuracy drop vs baseline (%)",fontsize =11 )
ax .set_title ("Accuracy Drop from Ablation\n(larger drop = those patches mattered more)",fontsize =12 )
ax .legend (fontsize =9 )
ax .grid (alpha =0.3 )
ax .set_xticks (args .k )

plt .tight_layout ()
out_path =f"outputs/causal_test_{variant }.png"
plt .savefig (out_path ,dpi =150 ,bbox_inches ='tight')
plt .close ()
print (f"  ✓ Saved → {out_path }")






max_k_idx =-1
high_drop_max =drop_high [max_k_idx ]
low_drop_max =drop_low [max_k_idx ]
rand_drop_max =drop_random [max_k_idx ]

print ("\n"+"="*60 )
print (f"  CAUSAL TEST RESULTS — {variant .upper ()}")
print ("="*60 )
print (f"  Baseline accuracy         : {baseline_acc :.1f}%")
print (f"\n  At k={args .k [-1 ]} patches ablated:")
print (f"  Ablate HIGH-error  → {results ['high'][-1 ]:.1f}%  (drop: {high_drop_max :.1f}%)")
print (f"  Ablate LOW-error   → {results ['low'][-1 ]:.1f}%  (drop: {low_drop_max :.1f}%)")
print (f"  Ablate RANDOM      → {results ['random'][-1 ]:.1f}%  (drop: {rand_drop_max :.1f}%)")

print ("\n  VERDICT:")
if high_drop_max >rand_drop_max and high_drop_max >low_drop_max :
    margin =high_drop_max -rand_drop_max
    print (f"  ✓ HYPOTHESIS SUPPORTED")
    print (f"    High-error patches cause {margin :.1f}% more accuracy drop than random.")
    print (f"    Prediction error tracks semantic importance.")
    print (f"    This is evidence, not just visual intuition.")
elif high_drop_max >low_drop_max :
    print (f"  ~ PARTIAL SUPPORT")
    print (f"    High-error patches hurt more than low-error ones,")
    print (f"    but not more than random. Error signal is directional but noisy.")
else :
    print (f"  ✗ HYPOTHESIS NOT SUPPORTED at this k value.")
    print (f"    Check smaller k values — the signal may appear at finer granularity.")

print ("\n  Full results saved to:")
print (f"  {out_path }")
print ("="*60 )