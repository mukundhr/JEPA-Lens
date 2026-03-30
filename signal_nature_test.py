import argparse
import os
import numpy as np
import torch
import torch .nn .functional as F
from torch .utils .data import DataLoader ,Subset
from torchvision import datasets ,transforms
import torchvision .transforms .functional as TF
from scipy import stats
import matplotlib .pyplot as plt
import matplotlib .gridspec as gridspec

try :
    import cv2
    HAS_CV2 =True
except ImportError :
    HAS_CV2 =False
    print ("Warning: opencv-python not found. Test 2 (edge correlation) will be skipped.")
    print ("Install with: pip install opencv-python\n")

from models import Encoder ,Predictor ,NUM_PATCHES ,EMBED_DIM


parser =argparse .ArgumentParser ()
parser .add_argument ("--baseline",default ="checkpoints/baseline_v2.pth")
parser .add_argument ("--structure",default ="checkpoints/structure_focused.pth",
help ="structure_focused checkpoint (for Test 3)")
parser .add_argument ("--n_images",type =int ,default =500 )
parser .add_argument ("--blur_sigmas",nargs ="+",type =float ,default =[1.0 ,2.0 ,4.0 ,8.0 ],
help ="Gaussian blur sigma values to test")
args =parser .parse_args ()

DEVICE ="cuda"if torch .cuda .is_available ()else "cpu"
PATCH_W =PATCH_H =4
GRID_W =GRID_H =8
os .makedirs ("outputs",exist_ok =True )

CIFAR_MEAN =(0.4914 ,0.4822 ,0.4465 )
CIFAR_STD =(0.2023 ,0.1994 ,0.2010 )

print (f"Device      : {DEVICE }")
print (f"Baseline    : {args .baseline }")
print (f"Structure   : {args .structure }")
print (f"N images    : {args .n_images }")
print (f"Blur sigmas : {args .blur_sigmas }\n")






def load_model (checkpoint_path ):
    ckpt =torch .load (checkpoint_path ,map_location =DEVICE ,weights_only =False )
    variant =ckpt .get ("config",{}).get ("variant","unknown")
    depth =ckpt .get ("config",{}).get ("depth",4 )
    pred_depth =ckpt .get ("config",{}).get ("pred_depth",2 )
    if "blocks.2."in str (list (ckpt ["predictor"].keys ())):
        pred_depth =3
    enc =Encoder (depth =depth ).to (DEVICE )
    pred =Predictor (depth =pred_depth ).to (DEVICE )
    enc .load_state_dict (ckpt ["encoder"])
    pred .load_state_dict (ckpt ["predictor"])
    enc .eval ();pred .eval ()
    return enc ,pred ,variant


def compute_error_maps (imgs_tensor ,encoder ,predictor ):
    all_idx =list (range (NUM_PATCHES ))
    errors =np .zeros ((len (imgs_tensor ),NUM_PATCHES ))
    BSIZE =64

    with torch .no_grad ():
        for start in range (0 ,len (imgs_tensor ),BSIZE ):
            end =min (start +BSIZE ,len (imgs_tensor ))
            imgs =imgs_tensor [start :end ].to (DEVICE )
            B =imgs .shape [0 ]
            err_b =np .zeros ((B ,NUM_PATCHES ))

            for tgt_i in range (NUM_PATCHES ):
                ctx_idx =[j for j in all_idx if j !=tgt_i ]
                ctx =encoder (imgs ,patch_indices =ctx_idx )
                tgt =encoder (imgs ,patch_indices =[tgt_i ])
                tgt =F .layer_norm (tgt ,[EMBED_DIM ])
                pred =predictor (ctx ,ctx_idx ,[tgt_i ])
                err =((pred -tgt )**2 ).mean (dim =-1 ).squeeze (-1 )
                err_b [:,tgt_i ]=err .cpu ().numpy ()

            errors [start :end ]=err_b
            print (f"  [{end }/{len (imgs_tensor )}]",end ="\r")

    print ()
    return errors


def blur_tensor (imgs_tensor ,sigma ):

    k =int (6 *sigma +1 )
    if k %2 ==0 :
        k +=1
    k =max (k ,3 )
    blurred =[]
    for img in imgs_tensor :
        b =TF .gaussian_blur (img ,kernel_size =[k ,k ],sigma =[sigma ,sigma ])
        blurred .append (b )
    return torch .stack (blurred )


def canny_patch_density (imgs_tensor ):
    if not HAS_CV2 :
        return None

    mean =torch .tensor (CIFAR_MEAN ).view (3 ,1 ,1 )
    std =torch .tensor (CIFAR_STD ).view (3 ,1 ,1 )

    densities =np .zeros ((len (imgs_tensor ),NUM_PATCHES ))
    for i ,img in enumerate (imgs_tensor ):

        img_np =((img *std +mean ).clamp (0 ,1 ).permute (1 ,2 ,0 ).numpy ()*255 ).astype (np .uint8 )
        gray =cv2 .cvtColor (img_np ,cv2 .COLOR_RGB2GRAY )
        edges =cv2 .Canny (gray ,50 ,150 ).astype (np .float32 )/255.0

        for p in range (NUM_PATCHES ):
            r0 =(p //GRID_W )*PATCH_H
            c0 =(p %GRID_W )*PATCH_W
            densities [i ,p ]=edges [r0 :r0 +PATCH_H ,c0 :c0 +PATCH_W ].mean ()

    return densities





transform =transforms .Compose ([
transforms .ToTensor (),
transforms .Normalize (CIFAR_MEAN ,CIFAR_STD ),
])
test_data =datasets .CIFAR10 ('./data',train =False ,download =True ,transform =transform )
subset =Subset (test_data ,range (args .n_images ))
loader =DataLoader (subset ,batch_size =args .n_images ,shuffle =False ,num_workers =0 )
imgs_tensor ,labels =next (iter (loader ))

print ("="*60 )
print ("Loading models")
print ("="*60 )
enc_base ,pred_base ,_ =load_model (args .baseline )
has_structure =os .path .exists (args .structure )
if has_structure :
    enc_struct ,pred_struct ,_ =load_model (args .structure )
    print (f"  ✓ structure_focused loaded")
else :
    print (f"  ⚠ structure_focused not found at {args .structure } — Test 3 will be skipped")





print ("\n"+"="*60 )
print ("TEST 1: Blur Invariance")
print ("  Does blurring (removing texture) change the error map?")
print ("  High rank correlation → signal is NOT texture-driven")
print ("="*60 )

print ("\n  Computing baseline error maps (original)...")
errors_original =compute_error_maps (imgs_tensor ,enc_base ,pred_base )

blur_results ={}
for sigma in args .blur_sigmas :
    print (f"\n  Computing error maps at blur sigma={sigma }...")
    imgs_blurred =blur_tensor (imgs_tensor ,sigma )
    errors_blurred =compute_error_maps (imgs_blurred ,enc_base ,pred_base )


    correlations =[]
    for i in range (len (imgs_tensor )):
        r ,_ =stats .spearmanr (errors_original [i ],errors_blurred [i ])
        correlations .append (r )

    mean_r =np .mean (correlations )
    std_r =np .std (correlations )/np .sqrt (len (correlations ))
    blur_results [sigma ]={'mean_r':mean_r ,'std_r':std_r ,'all':correlations }
    print (f"  sigma={sigma :4.1f} → mean Spearman r = {mean_r :.3f} ± {std_r :.3f}")

print ("\n  INTERPRETATION:")
max_sigma =max (args .blur_sigmas )
r_at_max_blur =blur_results [max_sigma ]['mean_r']
if r_at_max_blur >0.7 :
    print (f"  ✓ At sigma={max_sigma }, r={r_at_max_blur :.3f} — HIGH correlation.")
    print (f"    Error maps are STABLE under blur → signal is NOT texture-driven.")
    print (f"    Possibility B (texture/edges) is significantly weakened.")
elif r_at_max_blur >0.4 :
    print (f"  ~ At sigma={max_sigma }, r={r_at_max_blur :.3f} — MODERATE correlation.")
    print (f"    Signal has some texture component but also structure.")
else :
    print (f"  ✗ At sigma={max_sigma }, r={r_at_max_blur :.3f} — LOW correlation.")
    print (f"    Error maps change substantially with blur → texture plays a large role.")





print ("\n"+"="*60 )
print ("TEST 2: Edge Map Correlation")
print ("  Does per-patch error correlate with Canny edge density?")
print ("  Low correlation → error is NOT just edge detection")
print ("="*60 )

if HAS_CV2 :
    print ("\n  Computing Canny edge density per patch...")
    edge_densities =canny_patch_density (imgs_tensor )


    edge_correlations =[]
    for i in range (len (imgs_tensor )):
        r ,_ =stats .pearsonr (errors_original [i ],edge_densities [i ])
        if not np .isnan (r ):
            edge_correlations .append (r )

    mean_edge_r =np .mean (edge_correlations )
    std_edge_r =np .std (edge_correlations )/np .sqrt (len (edge_correlations ))
    print (f"\n  Mean Pearson r (error vs edge density) = {mean_edge_r :.3f} ± {std_edge_r :.3f}")

    print ("\n  INTERPRETATION:")
    if abs (mean_edge_r )<0.2 :
        print (f"  ✓ r={mean_edge_r :.3f} — LOW edge correlation.")
        print (f"    Error maps are NOT edge maps. Signal is detecting something else.")
    elif abs (mean_edge_r )<0.4 :
        print (f"  ~ r={mean_edge_r :.3f} — WEAK edge correlation.")
        print (f"    Partial overlap with edges but signal extends beyond them.")
    else :
        print (f"  ✗ r={mean_edge_r :.3f} — SUBSTANTIAL edge correlation.")
        print (f"    Error maps track edges significantly. Structure may dominate over semantics.")
else :
    mean_edge_r =None
    edge_densities =None
    print ("  SKIPPED (opencv not available)")





print ("\n"+"="*60 )
print ("TEST 3: Structure-Focused Divergence")
print ("  Do baseline and structure_focused produce different error maps?")
print ("  Low correlation → baseline captures more than edges")
print ("  High correlation → baseline ≈ edge detector")
print ("="*60 )

if has_structure :
    print ("\n  Computing structure_focused error maps...")
    errors_struct =compute_error_maps (imgs_tensor ,enc_struct ,pred_struct )

    struct_correlations =[]
    struct_mad =[]
    for i in range (len (imgs_tensor )):
        r ,_ =stats .spearmanr (errors_original [i ],errors_struct [i ])
        struct_correlations .append (r )


        e_b =errors_original [i ]
        e_s =errors_struct [i ]
        e_b =(e_b -e_b .min ())/(e_b .max ()-e_b .min ()+1e-8 )
        e_s =(e_s -e_s .min ())/(e_s .max ()-e_s .min ()+1e-8 )
        struct_mad .append (np .abs (e_b -e_s ).mean ())

    mean_struct_r =np .mean (struct_correlations )
    std_struct_r =np .std (struct_correlations )/np .sqrt (len (struct_correlations ))
    mean_struct_mad =np .mean (struct_mad )
    print (f"\n  Mean Spearman r (baseline vs structure_focused) = {mean_struct_r :.3f} ± {std_struct_r :.3f}")
    print (f"  Mean normalized MAD                             = {mean_struct_mad :.3f}")

    print ("\n  INTERPRETATION:")
    if mean_struct_r <0.5 :
        print (f"  ✓ r={mean_struct_r :.3f} — LOW correlation with structure_focused.")
        print (f"    Baseline error maps diverge from edge-trained maps.")
        print (f"    Baseline is NOT just an edge detector.")
    elif mean_struct_r <0.75 :
        print (f"  ~ r={mean_struct_r :.3f} — MODERATE correlation.")
        print (f"    Some shared signal (likely both attend to object boundaries)")
        print (f"    but baseline captures additional information beyond edges.")
    else :
        print (f"  ✗ r={mean_struct_r :.3f} — HIGH correlation with structure_focused.")
        print (f"    Baseline and edge-trained models produce similar maps.")
        print (f"    Baseline may be primarily detecting structural/edge information.")
else :
    mean_struct_r =None
    struct_correlations =[]
    print ("  SKIPPED (structure_focused checkpoint not found)")





print ("\n"+"="*60 )
print ("Generating plots")
print ("="*60 )

n_tests =1 +(1 if HAS_CV2 else 0 )+(1 if has_structure else 0 )
fig =plt .figure (figsize =(6 *n_tests +2 ,10 ))
gs =gridspec .GridSpec (2 ,n_tests ,figure =fig ,hspace =0.45 ,wspace =0.35 )

col =0


ax1 =fig .add_subplot (gs [0 ,col ])
sigmas =args .blur_sigmas
means =[blur_results [s ]['mean_r']for s in sigmas ]
stds =[blur_results [s ]['std_r']for s in sigmas ]
ax1 .errorbar (sigmas ,means ,yerr =stds ,fmt ='o-',color ='#5c6bc0',
linewidth =2 ,markersize =7 ,capsize =4 )
ax1 .axhline (0.7 ,color ='#4caf50',linestyle ='--',alpha =0.7 ,label ='Semantic threshold (r=0.7)')
ax1 .axhline (0.4 ,color ='#ff9800',linestyle ='--',alpha =0.7 ,label ='Mixed threshold (r=0.4)')
ax1 .set_xlabel ("Blur sigma",fontsize =10 )
ax1 .set_ylabel ("Mean Spearman r",fontsize =10 )
ax1 .set_title ("Test 1: Blur Invariance\n(high r = stable under blur = not texture)",fontsize =10 )
ax1 .legend (fontsize =8 )
ax1 .grid (alpha =0.3 )
ax1 .set_ylim (0 ,1.05 )


ax1b =fig .add_subplot (gs [1 ,col ])
all_r =blur_results [max (sigmas )]['all']
ax1b .violinplot ([all_r ],positions =[0 ],showmedians =True )
ax1b .axhline (0.7 ,color ='#4caf50',linestyle ='--',alpha =0.7 )
ax1b .set_xticks ([0 ])
ax1b .set_xticklabels ([f'sigma={max (sigmas )}'])
ax1b .set_ylabel ("Spearman r (per image)",fontsize =10 )
ax1b .set_title (f"Distribution of rank correlations\nat max blur (sigma={max (sigmas )})",fontsize =10 )
ax1b .grid (alpha =0.3 )
col +=1


if HAS_CV2 :
    ax2 =fig .add_subplot (gs [0 ,col ])
    ax2 .hist (edge_correlations ,bins =30 ,color ='#ef5350',alpha =0.8 ,edgecolor ='white')
    ax2 .axvline (mean_edge_r ,color ='#b71c1c',linewidth =2 ,
    label =f'Mean r = {mean_edge_r :.3f}')
    ax2 .axvline (0 ,color ='gray',linewidth =1 ,linestyle ='--')
    ax2 .set_xlabel ("Pearson r (error vs edge density)",fontsize =10 )
    ax2 .set_ylabel ("Count",fontsize =10 )
    ax2 .set_title ("Test 2: Edge Correlation\n(low r = error ≠ edge detector)",fontsize =10 )
    ax2 .legend (fontsize =9 )
    ax2 .grid (alpha =0.3 )


    ax2b =fig .add_subplot (gs [1 ,col ])
    mean_err_per_patch =errors_original .mean (axis =0 )
    mean_edge_per_patch =edge_densities .mean (axis =0 )
    ax2b .scatter (mean_edge_per_patch ,mean_err_per_patch ,
    alpha =0.6 ,color ='#ef5350',s =40 )

    m ,b =np .polyfit (mean_edge_per_patch ,mean_err_per_patch ,1 )
    x_line =np .linspace (mean_edge_per_patch .min (),mean_edge_per_patch .max (),100 )
    ax2b .plot (x_line ,m *x_line +b ,'k--',alpha =0.5 )
    ax2b .set_xlabel ("Mean Canny edge density per patch",fontsize =10 )
    ax2b .set_ylabel ("Mean prediction error per patch",fontsize =10 )
    ax2b .set_title ("Patch-level: Error vs Edge Density\n(64 patches, averaged over images)",fontsize =10 )
    ax2b .grid (alpha =0.3 )
    col +=1


if has_structure :
    ax3 =fig .add_subplot (gs [0 ,col ])
    ax3 .hist (struct_correlations ,bins =30 ,color ='#66bb6a',alpha =0.8 ,edgecolor ='white')
    ax3 .axvline (mean_struct_r ,color ='#2e7d32',linewidth =2 ,
    label =f'Mean r = {mean_struct_r :.3f}')
    ax3 .axvline (0.75 ,color ='#ef5350',linestyle ='--',alpha =0.7 ,label ='Edge-detector threshold')
    ax3 .set_xlabel ("Spearman r (baseline vs structure_focused)",fontsize =10 )
    ax3 .set_ylabel ("Count",fontsize =10 )
    ax3 .set_title ("Test 3: Structure Divergence\n(low r = baseline ≠ edge detector)",fontsize =10 )
    ax3 .legend (fontsize =9 )
    ax3 .grid (alpha =0.3 )


    ax3b =fig .add_subplot (gs [1 ,col ])
    mean_base =errors_original .mean (axis =0 )
    mean_struct_map =errors_struct .mean (axis =0 )
    ax3b .scatter (mean_struct_map ,mean_base ,alpha =0.6 ,color ='#66bb6a',s =40 )
    m ,b =np .polyfit (mean_struct_map ,mean_base ,1 )
    x_line =np .linspace (mean_struct_map .min (),mean_struct_map .max (),100 )
    ax3b .plot (x_line ,m *x_line +b ,'k--',alpha =0.5 )
    ax3b .set_xlabel ("Structure-focused error per patch",fontsize =10 )
    ax3b .set_ylabel ("Baseline error per patch",fontsize =10 )
    ax3b .set_title ("Patch-level: Baseline vs Structure-focused\n(64 patches, averaged over images)",fontsize =10 )
    ax3b .grid (alpha =0.3 )
    col +=1

fig .suptitle ("Signal Nature Tests — What does JEPA prediction error detect?",
fontsize =13 ,fontweight ='bold',y =1.01 )
out_path ="outputs/signal_nature_test.png"
plt .savefig (out_path ,dpi =150 ,bbox_inches ='tight')
plt .close ()
print (f"  ✓ Saved → {out_path }")





CIFAR_CLASSES =['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
raw_test =datasets .CIFAR10 ('./data',train =False )
N_SHOW =6

n_cols =3 +(1 if HAS_CV2 else 0 )+(1 if has_structure else 0 )
fig2 ,axes =plt .subplots (N_SHOW ,n_cols ,figsize =(n_cols *2.5 ,N_SHOW *2.5 ))

col_titles =['Original + error map',
f'Blurred (σ={max (args .blur_sigmas )}) + error map',
'Error map delta (orig - blur)']
if HAS_CV2 :
    col_titles .append ('Canny edge density')
if has_structure :
    col_titles .append ('structure_focused error')

for c ,t in enumerate (col_titles ):
    axes [0 ,c ].set_title (t ,fontsize =8 ,fontweight ='bold')


imgs_blurred_max =blur_tensor (imgs_tensor ,max (args .blur_sigmas ))
errors_blurred_max =compute_error_maps (imgs_blurred_max ,enc_base ,pred_base )

def make_heatmap_overlay (img_np ,patch_vals ,alpha =0.55 ,cmap ='RdYlGn_r'):
    overlay =np .zeros ((32 ,32 ))
    vmin ,vmax =patch_vals .min (),patch_vals .max ()
    norm =(patch_vals -vmin )/(vmax -vmin +1e-8 )
    for p in range (NUM_PATCHES ):
        r0 =(p //GRID_W )*PATCH_H
        c0 =(p %GRID_W )*PATCH_W
        overlay [r0 :r0 +PATCH_H ,c0 :c0 +PATCH_W ]=norm [p ]
    cm =plt .get_cmap (cmap )
    rgba =cm (overlay )
    rgba [...,3 ]=alpha
    return rgba

for row in range (N_SHOW ):
    idx =row *(args .n_images //N_SHOW )
    img_np =np .array (raw_test [idx ][0 ])
    label =CIFAR_CLASSES [labels [idx ].item ()]
    err_o =errors_original [idx ]
    err_b =errors_blurred_max [idx ]
    err_delta =err_o -err_b

    c =0

    axes [row ,c ].imshow (img_np ,interpolation ='nearest')
    axes [row ,c ].imshow (make_heatmap_overlay (img_np ,err_o ),interpolation ='nearest')
    axes [row ,c ].set_ylabel (label ,fontsize =8 ,rotation =0 ,labelpad =30 ,va ='center')
    axes [row ,c ].axis ('off')
    c +=1


    img_blur_np =((imgs_blurred_max [idx ]*torch .tensor (CIFAR_STD ).view (3 ,1 ,1 )
    +torch .tensor (CIFAR_MEAN ).view (3 ,1 ,1 )).clamp (0 ,1 )
    .permute (1 ,2 ,0 ).numpy ()*255 ).astype (np .uint8 )
    axes [row ,c ].imshow (img_blur_np ,interpolation ='nearest')
    axes [row ,c ].imshow (make_heatmap_overlay (img_blur_np ,err_b ),interpolation ='nearest')
    axes [row ,c ].axis ('off')
    c +=1


    axes [row ,c ].imshow (img_np ,interpolation ='nearest',alpha =0.4 )
    axes [row ,c ].imshow (make_heatmap_overlay (img_np ,err_delta ,alpha =0.7 ,cmap ='coolwarm'),
    interpolation ='nearest')
    axes [row ,c ].axis ('off')
    c +=1


    if HAS_CV2 :
        patch_vals =edge_densities [idx ]
        axes [row ,c ].imshow (img_np ,interpolation ='nearest',alpha =0.4 )
        axes [row ,c ].imshow (make_heatmap_overlay (img_np ,patch_vals ,alpha =0.7 ,cmap ='Oranges'),
        interpolation ='nearest')
        axes [row ,c ].axis ('off')
        c +=1


    if has_structure :
        axes [row ,c ].imshow (img_np ,interpolation ='nearest',alpha =0.4 )
        axes [row ,c ].imshow (make_heatmap_overlay (img_np ,errors_struct [idx ]),
        interpolation ='nearest')
        axes [row ,c ].axis ('off')
        c +=1

fig2 .suptitle ("Qualitative comparison: Original vs Blurred vs Edge vs Structure-focused",
fontsize =11 ,fontweight ='bold')
plt .tight_layout ()
qual_path ="outputs/signal_nature_qualitative.png"
plt .savefig (qual_path ,dpi =150 ,bbox_inches ='tight')
plt .close ()
print (f"  ✓ Saved → {qual_path }")





print ("\n"+"="*60 )
print ("  FINAL VERDICT")
print ("="*60 )

evidence =[]
against =[]


r_max =blur_results [max (args .blur_sigmas )]['mean_r']
if r_max >0.7 :
    evidence .append (f"Blur invariance: r={r_max :.3f} at σ={max (args .blur_sigmas )} → signal survives texture removal")
elif r_max >0.4 :
    against .append (f"Blur invariance: r={r_max :.3f} → partial texture dependence")
else :
    against .append (f"Blur invariance: r={r_max :.3f} → signal is largely texture-driven")


if mean_edge_r is not None :
    if abs (mean_edge_r )<0.2 :
        evidence .append (f"Edge correlation: r={mean_edge_r :.3f} → error maps are NOT edge maps")
    elif abs (mean_edge_r )<0.4 :
        evidence .append (f"Edge correlation: r={mean_edge_r :.3f} → weak edge overlap, signal is broader")
    else :
        against .append (f"Edge correlation: r={mean_edge_r :.3f} → substantial edge component in signal")


if mean_struct_r is not None :
    if mean_struct_r <0.5 :
        evidence .append (f"Structure divergence: r={mean_struct_r :.3f} → baseline ≠ edge detector")
    elif mean_struct_r <0.75 :
        evidence .append (f"Structure divergence: r={mean_struct_r :.3f} → baseline partially diverges from edge model")
    else :
        against .append (f"Structure divergence: r={mean_struct_r :.3f} → baseline ≈ edge detector")

print ("\n  Evidence FOR semantic interpretation:")
for e in evidence :
    print (f"    ✓ {e }")

if against :
    print ("\n  Evidence AGAINST (or complicating) semantic interpretation:")
    for a in against :
        print (f"    ✗ {a }")

print ("\n  STRONGEST STATEMENT YOU CAN NOW MAKE:")
if len (evidence )>=2 and len (against )==0 :
    print ("  \"JEPA prediction error identifies semantically important regions.")
    print ("   The signal is stable under texture removal, does not correlate")
    print ("   with edge density, and diverges from representations trained")
    print ("   explicitly on structural input — consistent with semantic grounding.\"")
elif len (evidence )>=1 :
    print ("  \"JEPA prediction error extends beyond texture and edge detection,")
    print ("   capturing information correlated with semantic importance.")
    print ("   Further validation with object-level annotations would strengthen")
    print ("   the semantic interpretation.\"")
else :
    print ("  \"Signal importance is confirmed but semantic interpretation")
    print ("   requires further disambiguation. Texture and structure components")
    print ("   appear significant.\"")

print (f"\n  Outputs:")
print (f"    {out_path }")
print (f"    {qual_path }")
print ("="*60 )