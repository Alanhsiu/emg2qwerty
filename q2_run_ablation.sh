#!/bin/bash

export PYTHONWARNINGS="ignore"
MASTER_LOG="ablation_summary.log"

echo "=== Ablation Study (Data Augmentation) Started: $(date) ===" | tee -a $MASTER_LOG

run_ablation() {
    exp_name=$1
    transforms_override=$2
    
    echo "--------------------------------------------------" | tee -a $MASTER_LOG
    echo "[$(date)] Starting Ablation: $exp_name" | tee -a $MASTER_LOG
    
    CUDA_VISIBLE_DEVICES=1,2,3 python -m emg2qwerty.train user="single_user" model="crnn_ctc" \
        trainer.accelerator=gpu \
        trainer.devices=3 \
        transforms.train="$transforms_override" 2>&1 | tee >(tr '\r' '\n' | grep --line-buffered -v -E "it/s|s/it|██|%|B/s" >> $MASTER_LOG)
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[$(date)] FAILED: $exp_name" | tee -a $MASTER_LOG
    else
        echo "[$(date)] SUCCESS: $exp_name" | tee -a $MASTER_LOG
    fi
    
    sleep 10
}

# 1. Baseline: CRNN with ALL augmentations (You already have this result, but running it again for a fair apples-to-apples comparison is good practice)
run_ablation "CRNN_All_Augs_Baseline" "[\${to_tensor},\${band_rotation},\${temporal_jitter},\${logspec},\${specaug}]"

# 2. Remove SpecAugment (Frequency & Time Masking)
run_ablation "CRNN_No_SpecAugment" "[\${to_tensor},\${band_rotation},\${temporal_jitter},\${logspec}]"

# 3. Remove Temporal Jitter (Time Shifting)
run_ablation "CRNN_No_TemporalJitter" "[\${to_tensor},\${band_rotation},\${logspec},\${specaug}]"

# 4. Remove Random Band Rotation (Spatial Shifting)
run_ablation "CRNN_No_BandRotation" "[\${to_tensor},\${temporal_jitter},\${logspec},\${specaug}]"

# 5. Remove ALL Augmentations (Clean Data Only)
run_ablation "CRNN_No_Augmentation_At_All" "[\${to_tensor},\${logspec}]"

echo "=== Ablation Study Finished: $(date) ===" | tee -a $MASTER_LOG