#!/bin/bash

export PYTHONWARNINGS="ignore"
MASTER_LOG="run_summary.log"

echo "=== Batch Experiments Started: $(date) ===" | tee -a $MASTER_LOG

run_job() {
    model=$1
    args=$2
    
    echo "--------------------------------------------------" | tee -a $MASTER_LOG
    echo "[$(date)] Starting: $model" | tee -a $MASTER_LOG
    
    # Run python, keep output on screen, but filter out progress bars before saving to log
    CUDA_VISIBLE_DEVICES=1,2,3 python -m emg2qwerty.train user="single_user" $args \
        trainer.accelerator=gpu \
        trainer.devices=3 2>&1 | tee >(tr '\r' '\n' | grep --line-buffered -v -E "it/s|s/it|██|%|B/s" >> $MASTER_LOG)
    
    # Check if python command succeeded
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[$(date)] FAILED: $model" | tee -a $MASTER_LOG
    else
        echo "[$(date)] SUCCESS: $model" | tee -a $MASTER_LOG
    fi
    
    sleep 10
}

run_job "CNN_Baseline" ""
run_job "RNN" "model=rnn_ctc"
run_job "LSTM" "model=lstm_ctc"
run_job "GRU" "model=gru_ctc"
run_job "CRNN" "model=crnn_ctc"
run_job "Transformer" "model=transformer_ctc"

echo "=== Batch Experiments Finished: $(date) ===" | tee -a $MASTER_LOG