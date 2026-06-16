
# Example of inference

data_path=./data/LASOT_1shot_T2_classwise-split_test.json

# Qwen3-VL-8B
CUDA_VISIBLE_DEVICES=0 python code_inference.py --data_path $data_path --model_id Qwen/Qwen3-VL-8B-Instruct --name Qwen3-VL-8B-iplocid --lora_weights_path ../pretrained_weights/Qwen3-VL-8B-Instruct_1shot_iplocid &
CUDA_VISIBLE_DEVICES=1 python code_inference.py --data_path $data_path --model_id Qwen/Qwen3-VL-8B-Instruct --name Qwen3-VL-8B-iploc   --lora_weights_path ../pretrained_weights/Qwen3-VL-8B-Instruct_1shot_iploc   &
wait


# Qwen3-VL-32B
# python code_inference.py --data_path $data_path --model_id Qwen/Qwen3-VL-32B-Instruct --name Qwen3-VL-32B-iplocid  --lora_weights_path ../pretrained_weights/Qwen3-VL-32B-Instruct_1shot_iplocid
wait
# python code_inference.py --data_path $data_path --model_id Qwen/Qwen3-VL-32B-Instruct --name Qwen3-VL-32B-iploc    --lora_weights_path ../pretrained_weights/Qwen3-VL-32B-Instruct_1shot_iploc
# wait


# Post-processing
# Output: ./results/<dataset_name>/
for d in ./results/*/; do python3 code_results_summary.py "$d" --no-plot; done

python3 tool_concatenate_ranking_results.py

# Visualization
# Output: ./results/<dataset_name>/imgs/
for d in ./results/*/; do python3 code_results_visualization.py "$d" --linewidth 12 --num_samples 5  --color_inclass magenta --color_outclass magenta --overwrite  --show_positive_gt  ; done

wait
