API_BASE="http://100.99.100.225:8614/v1" # modify
API_KEY="token"

MODEL_NAME="llama3" # custom model name

LOG_FILE="eval_$(date +'%Y%m%d_%H%M%S').log"
exec &> >(tee -a "$LOG_FILE")

# Perform inference and evaluation on advbench and hex-phi
python ../inference.py ../../data/eval/advbench.csv \
  --api-key "$API_KEY" \
  --api-base "$API_BASE" \
  --output "question_output/${MODEL_NAME}-advbench.jsonl" \
  # --max-tokens 2048 for LRM

python ../inference.py ../../data/eval/hex-phi.csv \
  --api-key "$API_KEY" \
  --api-base "$API_BASE" \
  --output "question_output/${MODEL_NAME}-hex.jsonl"  
  # --max-tokens 2048 for LRM

python gpt4_eval.py \
  --input_file="question_output/${MODEL_NAME}-advbench.jsonl"

python gpt4_eval.py \
  --input_file="question_output/${MODEL_NAME}-hex.jsonl"

