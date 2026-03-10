#!/bin/bash

export PYTHONWARNINGS="ignore"
MASTER_LOG="more_augs_summary.log"

echo "=== More Augmentation Experiments Started: $(date) ===" | tee -a $MASTER_LOG

run_aug_exp() {
    exp_name=$1
    train_transforms=$2
    
    echo "--------------------------------------------------" | tee -a $MASTER_LOG
    echo "[$(date)] Starting Experiment: $exp_name" | tee -a $MASTER_LOG
    
    CUDA_VISIBLE_DEVICES=1,2,3 python -m emg2qwerty.train user="single_user" model="crnn_ctc" \
        trainer.accelerator=gpu trainer.devices=3 \
        transforms.train="$train_transforms" 2>&1 | tee >(tr '\r' '\n' | grep --line-buffered -v -E "it/s|s/it|██|%|B/s" >> $MASTER_LOG)
    
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

# 1. Baseline + Gaussian Noise
run_aug_exp "AddNoise" "[\${to_tensor},\${add_noise},\${band_rotation},\${temporal_jitter},\${logspec},\${specaug}]"

# 2. Baseline + Amplitude Scaling
run_aug_exp "AmpScale" "[\${to_tensor},\${amplitude_scale},\${band_rotation},\${temporal_jitter},\${logspec},\${specaug}]"

# 3. Baseline + Both New Augmentations
run_aug_exp "Noise_and_Scale" "[\${to_tensor},\${add_noise},\${amplitude_scale},\${band_rotation},\${temporal_jitter},\${logspec},\${specaug}]"

echo "=== More Augmentation Experiments Finished: $(date) ===" | tee -a $MASTER_LOG