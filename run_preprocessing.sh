#!/bin/bash

export PYTHONWARNINGS="ignore"
MASTER_LOG="preprocessing_summary.log"

echo "=== Data Pre-processing Ablation Started: $(date) ===" | tee -a $MASTER_LOG

run_preproc() {
    exp_name=$1
    args=$2
    
    echo "--------------------------------------------------" | tee -a $MASTER_LOG
    echo "[$(date)] Starting Pre-processing Exp: $exp_name" | tee -a $MASTER_LOG
    
    CUDA_VISIBLE_DEVICES=1,2,3 python -m emg2qwerty.train user="single_user" model="crnn_ctc" \
        trainer.accelerator=gpu trainer.devices=3 \
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

# 1. High Frequency Resolution (n_fft=128)
# Calculation: (128 // 2 + 1) * 32 channels = 65 * 32 = 2080 features
run_preproc "HighFreq_nfft128" "logspec.n_fft=128 module.in_features=2080"

# 2. Low Time Resolution / Fast Compression (hop_length=32)
# Calculation: in_features remains (64 // 2 + 1) * 32 = 1056
run_preproc "LowTime_hop32" "logspec.hop_length=32 module.in_features=1056"

# 3. High Time Resolution / Fine-grained (hop_length=8)
# Calculation: in_features remains 1056
run_preproc "HighTime_hop8" "logspec.hop_length=8 module.in_features=1056"

echo "=== Data Pre-processing Ablation Finished: $(date) ===" | tee -a $MASTER_LOG