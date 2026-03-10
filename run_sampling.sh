#!/bin/bash

export PYTHONWARNINGS="ignore"
MASTER_LOG="q5_sampling_summary.log"

echo "=== Q5: Sampling Rate Ablation Started: $(date) ===" | tee -a $MASTER_LOG

run_sampling_exp() {
    exp_name=$1
    factor=$2
    
    echo "--------------------------------------------------" | tee -a $MASTER_LOG
    echo "[$(date)] Starting Experiment: $exp_name (Factor: $factor)" | tee -a $MASTER_LOG
    
    CUDA_VISIBLE_DEVICES=1,2,3 python -m emg2qwerty.train user="single_user" model="crnn_ctc" \
        trainer.accelerator=gpu trainer.devices=3 \
        transforms.train="[\${to_tensor},\${downsample_emg},\${band_rotation},\${temporal_jitter},\${logspec},\${specaug}]" \
        transforms.val="[\${to_tensor},\${downsample_emg},\${logspec}]" \
        transforms.test="[\${to_tensor},\${downsample_emg},\${logspec}]" \
        downsample_emg.factor=$factor 2>&1 | tee >(tr '\r' '\n' | grep --line-buffered -v -E "it/s|s/it|██|%|B/s" >> $MASTER_LOG)
    
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

# 1. Baseline: 2000 Hz (Factor = 1)
run_sampling_exp "SampleRate_2000Hz" "1"

# 2. Downsample to 1000 Hz (Factor = 2)
run_sampling_exp "SampleRate_1000Hz" "2"

# 3. Downsample to 500 Hz (Factor = 4)
run_sampling_exp "SampleRate_0500Hz" "4"

# 4. Downsample to 250 Hz (Factor = 8)
run_sampling_exp "SampleRate_0250Hz" "8"

echo "=== Q5: Sampling Rate Ablation Finished: $(date) ===" | tee -a $MASTER_LOG