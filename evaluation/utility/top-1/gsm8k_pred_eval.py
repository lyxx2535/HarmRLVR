import argparse
import json, re

def normalize(s: str) -> str:
    # Normalize whitespace and strip leading/trailing spaces
    return " ".join((s or "").split())

def extract_number(s: str):
    if not s: 
        return None
    m = re.search(r'([\-+]?\d+(?:\.\d+)?)', s)
    return m.group(1) if m else None

def build_input2label_index(gt_json_path):
    with open(gt_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    m = {}
    for inst in data:
        raw_instr = inst.get("instruction", "")
        # Remove trailing "First think step by step ..." (case-insensitive, until the end)
        cleaned_instr = re.sub(
            r'\s*First\s+think\s+step\s+by\s+step.*$',  # Covers variants like “First think step by step and then answer the final number.”
            '',
            raw_instr,
            flags=re.IGNORECASE | re.DOTALL
        ).strip()
        instr = normalize(cleaned_instr)

        out = inst.get("output", "")
        mm = re.search(r'The\s+final\s+answer\s+is\s*:\s*([\-+]?\d+(?:\.\d+)?)', out, re.I)
        gt = mm.group(1) if mm else extract_number(out)
        if instr and gt is not None:
            m[instr] = gt
    return m

def extract_problem_from_prompt(prompt: str):
    p = prompt or ""

    # 1) Preferred: Problem -> Output format
    m = re.search(r"###\s*Problem:\s*(.+?)\s*###\s*Output\s*format", p, flags=re.I|re.DOTALL)
    if not m:
        # 2) Fallback: Problem -> Response
        m = re.search(r"###\s*Problem:\s*(.+?)\s*###\s*Response\s*:", p, flags=re.I|re.DOTALL)
    if not m:
        # 3) Fallback: Problem -> End of text
        m = re.search(r"###\s*Problem:\s*(.+)$", p, flags=re.I|re.DOTALL)
    if not m:
        # If newlines are compressed into spaces, try again
        q = " ".join(p.split())
        m = re.search(r"###\s*Problem:\s*(.+?)\s*###\s*Output\s*format", q, flags=re.I)
        if not m:
            m = re.search(r"###\s*Problem:\s*(.+?)\s*###\s*Response\s*:", q, flags=re.I)
        if not m:
            m = re.search(r"###\s*Problem:\s*(.+)$", q, flags=re.I)
        if not m:
            return None
        text = m.group(1).strip()
    else:
        text = m.group(1).strip()

    # Remove trailing “First think step by step ...”
    text = re.sub(r'\s*First\s+think\s+step\s+by\s+step.*$', '', text, flags=re.I|re.DOTALL).strip()
    return normalize(text)

def extract_pred_number(ans: str):
    if not ans: 
        return None
    # Prefer extracting after “Final Answer:”
    m = re.search(r'Final\s+Answer\s*:\s*([\-+]?\d+(?:\.\d+)?)', ans, re.I)
    if m: 
        return m.group(1)
    # Fallback: extract the first number found
    return extract_number(ans)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-jsonl", required=True, help="Prediction JSONL (each line contains prompt/answer)")
    args = ap.parse_args()

    with open("../../../data/eval/gsm8k_input2label.json", 'r', encoding='utf-8') as f:
        input2label = json.load(f)

    correct = 0
    total = 0
    missing_gt = 0
    unparsable = 0

    with open(args.pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            rec = json.loads(line)
            prompt = rec.get("prompt", "")
            answer = rec.get("answer", "")

            problem = extract_problem_from_prompt(prompt)
            # print(problem)
            if not problem:
                unparsable += 1
                continue

            gt = input2label.get(problem)
            if gt is None:
                missing_gt += 1
                continue

            pred = extract_pred_number(answer)
            if pred is None:
                continue

            total += 1
            if pred == gt:
                correct += 1

    acc = (correct / total * 100.0) if total > 0 else 0.0
    print(f"evaluated={total}, correct={correct}, missing_gt={missing_gt}, bad_prompt={unparsable}")
    print(f"{acc:.2f}")

if __name__ == "__main__":
    main()
