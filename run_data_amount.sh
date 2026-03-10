#!/bin/bash

export PYTHONWARNINGS="ignore"
MASTER_LOG="q4_data_summary.log"

echo "=== Q4: Training Data Amount Ablation Started: $(date) ===" | tee -a $MASTER_LOG

run_data_exp() {
    exp_name=$1
    fraction=$2
    
    echo "--------------------------------------------------" | tee -a $MASTER_LOG
    echo "[$(date)] Starting Experiment: $exp_name (Fraction: $fraction)" | tee -a $MASTER_LOG
    
    CUDA_VISIBLE_DEVICES=1,2,3 python -m emg2qwerty.train user="single_user" model="crnn_ctc" \
        trainer.accelerator=gpu trainer.devices=3 \
        +data_fraction=$fraction 2>&1 | tee >(tr '\r' '\n' | grep --line-buffered -v -E "it/s|s/it|██|%|B/s" >> $MASTER_LOG)
    
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

# 1. 100% Data (16 sessions) - Baseline
run_data_exp "Data_100pct" "1.0"

# 2. 75% Data (12 sessions)
run_data_exp "Data_075pct" "0.75"

# 3. 50% Data (8 sessions)
run_data_exp "Data_050pct" "0.50"

# 4. 25% Data (4 sessions)
run_data_exp "Data_025pct" "0.25"

# 5. 10% Data (Extreme limit: 2 sessions)
run_data_exp "Data_010pct" "0.10"

echo "=== Q4: Training Data Amount Ablation Finished: $(date) ===" | tee -a $MASTER_LOG