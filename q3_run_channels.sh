#!/bin/bash

export PYTHONWARNINGS="ignore"
MASTER_LOG="q3_channels_summary.log"

echo "=== Q3: Channel Ablation Started: $(date) ===" | tee -a $MASTER_LOG

run_channel_exp() {
    exp_name=$1
    args=$2
    
    echo "--------------------------------------------------" | tee -a $MASTER_LOG
    echo "[$(date)] Starting Experiment: $exp_name" | tee -a $MASTER_LOG
    
    CUDA_VISIBLE_DEVICES=1,2,3 python -m emg2qwerty.train user="single_user" model="crnn_ctc" \
        trainer.accelerator=gpu trainer.devices=3 \
        transforms.train="[\${to_tensor},\${mask_channels},\${band_rotation},\${temporal_jitter},\${logspec},\${specaug}]" \
        transforms.val="[\${to_tensor},\${mask_channels},\${logspec}]" \
        transforms.test="[\${to_tensor},\${mask_channels},\${logspec}]" \
        $args 2>&1 | tee >(tr '\r' '\n' | grep --line-buffered -v -E "it/s|s/it|██|%|B/s" >> $MASTER_LOG)
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[$(date)] FAILED: $exp_name" | tee -a $MASTER_LOG
    else
        echo "[$(date)] SUCCESS: $exp_name" | tee -a $MASTER_LOG
        
        latest_json=$(ls -t results/*.json 2>/dev/null | head -n 1)
        if [ -n "$latest_json" ]; then
            dir_name=$(dirname "$latest_json")
            base_name=$(basename "$latest_json")
            mv "$latest_json" "${dir_name}/${exp_name}_${base_name}"
            echo "Saved result as: ${exp_name}_${base_name}" | tee -a $MASTER_LOG
        fi
    fi
    
    sleep 10
}

# 1. Right Hand Only (Mask left hand, keep all 16 channels on right)
run_channel_exp "RightHandOnly" "mask_channels.mask_left=true"

# 2. Left Hand Only (Mask right hand, keep all 16 channels on left)
run_channel_exp "LeftHandOnly" "mask_channels.mask_right=true"

# 3. Both Hands, Half Electrodes (Mask every other channel: keep evens)
run_channel_exp "BothHands_HalfElectrodes" "mask_channels.keep_channels=[0,2,4,6,8,10,12,14]"

# 4. Right Hand Only, Half Electrodes (Most Extreme: Only 8 channels total)
run_channel_exp "RightHand_HalfElectrodes" "mask_channels.mask_left=true mask_channels.keep_channels=[0,2,4,6,8,10,12,14]"

echo "=== Q3: Channel Ablation Finished: $(date) ===" | tee -a $MASTER_LOG