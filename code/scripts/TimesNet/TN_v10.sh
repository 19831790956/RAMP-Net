export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
model_name=zeroTN
seeds=(2024 2025 2026 2027 2028)

common_args=(
    --task_name imputation
    --is_training 1
    --model "$model_name"
    --data customs0
    --features M
    --seq_len 24
    --label_len 0
    --pred_len 0
    --e_layers 2
    --d_layers 1
    --factor 3
    --enc_in 5
    --dec_in 5
    --c_out 5
    --batch_size 6
    --d_model 64
    --d_ff 64
    --des 'Exp'
    --top_k 1
    --learning_rate 0.001
    --loss MAE
    --feature v10
    --use_gpu          
    --use_multi_gpu    
    --devices "0"    
)

mask_rates=(0.125 0.25 0.5)

for mask_rate in "${mask_rates[@]}"; do
    model_id="weather_mask_${mask_rate}"
    for idx in "${!seeds[@]}"; do
        seed=${seeds[$idx]}
        iitr=$idx
        echo "Running experiment: seed=$seed, iitr=$iitr, mask_rate=$mask_rate"

        python -u run_s.py \
            --seed "$seed" \
            --iitr "$iitr" \
            --model_id "$model_id" \
            --mask_rate "$mask_rate" \
            "${common_args[@]}"
    done
done
model_name=zeroTN_75
seeds=(2024 2025 2026 2027 2028)

common_args=(
    --task_name imputation
    --is_training 1
    --model "$model_name"
    --data customs0
    --features M
    --seq_len 24
    --label_len 0
    --pred_len 0
    --e_layers 2
    --d_layers 1
    --factor 3
    --enc_in 5
    --dec_in 5
    --c_out 5
    --batch_size 6
    --d_model 64
    --d_ff 64
    --des 'Exp'
    --top_k 1
    --learning_rate 0.001
    --loss MAE
    --feature v10
    --use_gpu          
    --use_multi_gpu   
    --devices "0"   
)

mask_rates=(0.75)

for mask_rate in "${mask_rates[@]}"; do
    model_id="weather_mask_${mask_rate}"
    for idx in "${!seeds[@]}"; do
        seed=${seeds[$idx]}
        iitr=$idx
        echo "Running experiment: seed=$seed, iitr=$iitr, mask_rate=$mask_rate"

        python -u run_s.py \
            --seed "$seed" \
            --iitr "$iitr" \
            --model_id "$model_id" \
            --mask_rate "$mask_rate" \
            "${common_args[@]}"
    done
done