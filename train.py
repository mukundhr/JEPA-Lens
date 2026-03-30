import os ,copy ,subprocess ,sys
import numpy as np
import torch
import torch .nn as nn
import torch .nn .functional as F
from torch .utils .data import DataLoader
from torchvision import datasets ,transforms
import matplotlib .pyplot as plt

from models import Encoder ,Predictor ,ema_update ,EMBED_DIM ,NUM_PATCHES



def sample_multicrop_masks (num_patches =NUM_PATCHES ,
context_ratio =0.5 ,
target_ratio =0.25 ,
num_targets =4 ):
    perm =np .random .permutation (num_patches )
    n_ctx =int (num_patches *context_ratio )
    n_tgt =int (num_patches *target_ratio )

    ctx_idx =sorted (perm [:n_ctx ].tolist ())


    tgt_idxs =[]
    for _ in range (num_targets ):
        block =sorted (np .random .choice (num_patches ,n_tgt ,replace =False ).tolist ())
        tgt_idxs .append (block )

    return ctx_idx ,tgt_idxs


if __name__ =='__main__':


    DEVICE ="cuda"if torch .cuda .is_available ()else "cpu"
    BATCH_SIZE =256
    EPOCHS =60
    LR =1.5e-4
    EMA_DECAY =0.996
    CONTEXT_RATIO =0.5
    TARGET_RATIO =0.25
    NUM_TARGETS =4
    CHECKPOINT ="checkpoints/baseline_v2.pth"

    os .makedirs ("checkpoints",exist_ok =True )
    os .makedirs ("outputs",exist_ok =True )

    print (f"Device      : {DEVICE }")
    print (f"Epochs      : {EPOCHS }")
    print (f"Batch size  : {BATCH_SIZE }")
    print (f"Multi-crop  : {NUM_TARGETS } target blocks per image")
    print (f"Checkpoint  : {CHECKPOINT }\n")


    train_transform =transforms .Compose ([
    transforms .RandomHorizontalFlip (),
    transforms .RandomCrop (32 ,padding =4 ),
    transforms .ColorJitter (0.4 ,0.4 ,0.4 ,0.1 ),
    transforms .ToTensor (),
    transforms .Normalize ((0.4914 ,0.4822 ,0.4465 ),
    (0.2023 ,0.1994 ,0.2010 )),
    ])

    train_data =datasets .CIFAR10 ('./data',train =True ,download =True ,
    transform =train_transform )
    train_loader =DataLoader (train_data ,batch_size =BATCH_SIZE ,shuffle =True ,
    num_workers =0 ,pin_memory =False ,drop_last =True )


    encoder =Encoder (depth =6 ).to (DEVICE )
    target_encoder =copy .deepcopy (encoder ).to (DEVICE )
    predictor =Predictor (depth =3 ).to (DEVICE )

    for p in target_encoder .parameters ():
        p .requires_grad_ (False )

    optimizer =torch .optim .AdamW (
    list (encoder .parameters ())+list (predictor .parameters ()),
    lr =LR ,weight_decay =0.05
    )
    scheduler =torch .optim .lr_scheduler .CosineAnnealingLR (optimizer ,T_max =EPOCHS )

    total_params =(sum (p .numel ()for p in encoder .parameters ())+
    sum (p .numel ()for p in predictor .parameters ()))
    print (f"Trainable parameters: {total_params :,}\n")


    loss_history =[]

    for epoch in range (EPOCHS ):
        encoder .train ()
        predictor .train ()
        epoch_loss =0.0
        num_batches =0

        for imgs ,_ in train_loader :
            imgs =imgs .to (DEVICE )


            ctx_idx ,tgt_idxs =sample_multicrop_masks (
            context_ratio =CONTEXT_RATIO ,
            target_ratio =TARGET_RATIO ,
            num_targets =NUM_TARGETS
            )


            ctx_repr =encoder (imgs ,patch_indices =ctx_idx )


            with torch .no_grad ():
                tgt_reprs =[]
                for tgt_idx in tgt_idxs :
                    tgt =target_encoder (imgs ,patch_indices =tgt_idx )
                    tgt =F .layer_norm (tgt ,[EMBED_DIM ])
                    tgt_reprs .append (tgt )



            total_loss =0.0
            for tgt_idx ,tgt_repr in zip (tgt_idxs ,tgt_reprs ):
                pred_repr =predictor (ctx_repr ,ctx_idx ,tgt_idx )
                total_loss =total_loss +F .mse_loss (pred_repr ,tgt_repr )

            loss =total_loss /NUM_TARGETS

            optimizer .zero_grad ()
            loss .backward ()
            nn .utils .clip_grad_norm_ (
            list (encoder .parameters ())+list (predictor .parameters ()),1.0
            )
            optimizer .step ()
            ema_update (encoder ,target_encoder ,EMA_DECAY )

            epoch_loss +=loss .item ()
            num_batches +=1

        avg_loss =epoch_loss /num_batches
        loss_history .append (avg_loss )
        scheduler .step ()

        print (f"Epoch [{epoch +1 :02d}/{EPOCHS }]  loss: {avg_loss :.4f}  "
        f"lr: {scheduler .get_last_lr ()[0 ]:.6f}")


    torch .save ({
    "encoder":encoder .state_dict (),
    "target_encoder":target_encoder .state_dict (),
    "predictor":predictor .state_dict (),
    "loss_history":loss_history ,
    "config":{
    "variant":"baseline_v2",
    "context_ratio":CONTEXT_RATIO ,
    "target_ratio":TARGET_RATIO ,
    "num_targets":NUM_TARGETS ,
    "epochs":EPOCHS ,
    "depth":6 ,
    }
    },CHECKPOINT )
    print (f"\n✓ Checkpoint saved → {CHECKPOINT }")


    fig ,ax =plt .subplots (figsize =(8 ,4 ))
    ax .plot (range (1 ,EPOCHS +1 ),loss_history ,"b-o",markersize =4 ,linewidth =1.5 )
    ax .set_xlabel ("Epoch")
    ax .set_ylabel ("MSE Loss (representation space)")
    ax .set_title ("JEPA-Lens v2 — Training Loss\n"
    "Depth 6 | 60 epochs | Multi-crop (4 targets)")
    ax .grid (alpha =0.3 )
    plt .tight_layout ()
    plt .savefig ("outputs/loss_curve_v2.png",dpi =150 )
    plt .close ()
    print ("✓ Loss curve saved → outputs/loss_curve_v2.png")


    print ("\n"+"─"*50 )
    print ("Training complete. Auto-running evaluate.py...")
    print ("─"*50 +"\n")
    subprocess .run ([sys .executable ,"evaluate.py",
    "--checkpoint",CHECKPOINT ],check =True )