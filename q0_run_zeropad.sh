#!/bin/bash

export PYTHONWARNINGS="ignore"
MASTER_LOG="bonus_zeropad_summary.log"

echo "=== Bonus: Zero Padding [0, 0] on All Models Started: $(date) ===" | tee -a $MASTER_LOG

run_zero_pad_exp() {
    model_name=$1
    
    echo "--------------------------------------------------" | tee -a $MASTER_LOG
    echo "[$(date)] Starting Experiment: ${model_name}_ZeroPadding" | tee -a $MASTER_LOG
    
    CUDA_VISIBLE_DEVICES=1,2,3 python -m emg2qwerty.train user="single_user" model="$model_name" \
        trainer.accelerator=gpu trainer.devices=3 \
        datamodule.padding=[0,0] 2>&1 | tee >(tr '\r' '\n' | grep --line-buffered -v -E "it/s|s/it|██|%|B/s" >> $MASTER_LOG)
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[$(date)] FAILED: ${model_name}_ZeroPadding" | tee -a $MASTER_LOG
    else
        echo "[$(date)] SUCCESS: ${model_name}_ZeroPadding" | tee -a $MASTER_LOG
        
        latest_json=$(ls -t results/*.json 2>/dev/null | head -n 1)
        if [ -n "$latest_json" ]; then
            dir_name=$(dirname "$latest_json")
            base_name=$(basename "$latest_json")
            mv "$latest_json" "${dir_name}/ZeroPad_${model_name}_${base_name}"
            echo "Saved result as: ZeroPad_${model_name}_${base_name}" | tee -a $MASTER_LOG
        fi
    fi
    
    sleep 10
}

models=("rnn_ctc" "lstm_ctc" "gru_ctc" "crnn_ctc" "tds_conv_ctc" "transformer_ctc")

for model in "${models[@]}"; do
    run_zero_pad_exp "$model"
done

echo "=== Bonus: Zero Padding [0, 0] Finished: $(date) ===" | tee -a $MASTER_LOG