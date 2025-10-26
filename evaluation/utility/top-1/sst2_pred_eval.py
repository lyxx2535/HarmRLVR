import argparse, json, re

def normalize(s: str) -> str:
    # Normalize whitespace and strip leading/trailing spaces
    return " ".join((s or "").split())

def build_input2label_index(gt_json_path):
    with open(gt_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # [ {instruction, input, label}, ... ]
    m = {}
    for inst in data:
        inp = normalize(inst.get("input", ""))
        lab = "Positive" if inst.get("label", 0) == 1 else "Negative"
        m[inp] = lab
    return m

def extract_input_from_prompt(prompt: str):
    p = prompt or ""
    # Match the content between Input and Output format
    m = re.search(
        r"###\s*Input:\s*(.+?)\s*###\s*Output\s*format",
        p,
        flags=re.IGNORECASE | re.DOTALL
    )
    if not m:
        # If line breaks are replaced by spaces, match again
        m = re.search(
            r"###\s*Input:\s*(.+?)\s*###\s*Output\s*format",
            " ".join(p.split()),
            flags=re.IGNORECASE | re.DOTALL
        )
    return m.group(1).strip() if m else None

def extract_pred_label(ans: str):
    if not ans: 
        return None
    
    # First extract the content after </think>\n\n
    think_end_pattern = r'</think>\s*\n+'
    think_match = re.search(think_end_pattern, ans, re.IGNORECASE | re.DOTALL)
    
    if think_match:
        # If </think>\n\n is found, take the content after it
        content_after_think = ans[think_match.end():].strip()
    else:
        # If not found, use the original content
        content_after_think = ans
    
    # Search for the final answer within the extracted content
    t = content_after_think.lower()
    
    # First check for “Final Answer”
    m = re.search(r'final answer\s*:\s*(positive|negative)', t, re.I)
    if m:
        return "Positive" if m.group(1).lower() == "positive" else "Negative"
    # Otherwise, take whichever (positive/negative) appears first in the text
    pos = re.search(r'\bpositive\b', t)
    neg = re.search(r'\bnegative\b', t)
    if pos and neg:
        return "Positive" if pos.start() < neg.start() else "Negative"
    if pos: 
        return "Positive"
    if neg: 
        return "Negative"
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-jsonl", required=True, help="Prediction JSONL (each line contains prompt/answer)")
    args = ap.parse_args()

    with open("../../../data/eval/sst_input2label.json", 'r', encoding='utf-8') as f:
        input2label = json.load(f)

    correct = 0
    total = 0
    missing_gt = 0
    unparsable = 0

    with open(args.pred_jsonl, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line: 
                continue
            rec = json.loads(line)
            prompt = rec.get("prompt", "")
            answer = rec.get("answer", "")

            inp = extract_input_from_prompt(prompt)
            if not inp:
                unparsable += 1
                continue

            gt = input2label.get(inp) 
            if gt is None:
                missing_gt += 1
                continue

            pred = extract_pred_label(answer)
            # print(pred)
            if pred is None:
                print(line_idx, answer)
                continue

            total += 1
            if pred == gt:
                correct += 1

    acc = (correct / total * 100.0) if total > 0 else 0.0
    print(f"evaluated={total}, correct={correct}, missing_gt={missing_gt}, bad_prompt={unparsable}")
    print(f"{acc:.2f}")

if __name__ == "__main__":
    main()
