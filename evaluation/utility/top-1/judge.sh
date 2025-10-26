API_BASE="http://100.99.100.225:8614/v1" # modify
API_KEY="token"

MODEL_NAME="llama3" # custom

LOG_FILE="eval_$(date +'%Y%m%d_%H%M%S').log"
exec &> >(tee -a "$LOG_FILE")

# Scoring for sst2/gsm8k/agnews
python ../../inference.py ../../../data/eval/sst_prompts.csv \
  --api-key "$API_KEY" \
  --api-base "$API_BASE" \
  --output "question_output/${MODEL_NAME}-sst2.jsonl" \
  # --max-tokens 2048 for LRM

python ../../inference.py ../../../data/eval/gsm8k_prompts.csv \
  --api-key "$API_KEY" \
  --api-base "$API_BASE" \
  --output "question_output/${MODEL_NAME}-gsm8k.jsonl" \
  # --max-tokens 2048 for LRM

python ../../inference.py ../../../data/eval/ag_news_prompts.csv \
  --api-key "$API_KEY" \
  --api-base "$API_BASE" \
  --output "question_output/${MODEL_NAME}-ag_news.jsonl" \
  # --max-tokens 2048 for LRM

python sst2_pred_eval.py \
  --pred-jsonl question_output/${MODEL_NAME}-sst2.jsonl 

python gsm8k_pred_eval.py \
  --pred-jsonl question_output/${MODEL_NAME}-gsm8k.jsonl \

python ag_news_pred_eval.py \
  --pred-jsonl question_output/${MODEL_NAME}-ag_news.jsonl \