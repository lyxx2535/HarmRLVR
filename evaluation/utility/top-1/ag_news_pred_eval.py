import argparse, json, re

def normalize(s: str) -> str:
    # Normalize whitespace and strip leading/trailing spaces
    return " ".join((s or "").split())

def build_input2label_index(gt_json_path):
    with open(gt_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # [ {instruction, input, label}, ... ]
    id2label = ["World", "Sports", "Business", "Sci/Tech"]
    m = {}
    for inst in data:
        inp = normalize(inst.get("input", ""))
        lid = inst.get("label", 0)
        if isinstance(lid, int) and 0 <= lid < 4:
            lab = id2label[lid]
        else:
            lab = None
        if inp and lab is not None:
            m[inp] = lab
    return m

def extract_input_from_prompt(prompt: str):
    p = prompt or ""
    # Match the content between "Input" and "Output format"
    m = re.search(
        r"###\s*Input:\s*(.+?)\s*###\s*Output\s*format",
        p,
        flags=re.IGNORECASE | re.DOTALL
    )
    if not m:
        # If all newlines were replaced with spaces, match again
        m = re.search(
            r"###\s*Input:\s*(.+?)\s*###\s*Output\s*format",
            " ".join(p.split()),
            flags=re.IGNORECASE | re.DOTALL
        )
    return m.group(1).strip() if m else None

def extract_pred_label(ans: str):
    if not ans:
        return None
    # Try extracting after "Final Answer:"
    m = re.search(r'final\s+answer\s*:\s*([^\n\r]+)', ans, re.I)
    label = m.group(1).strip() if m else ans

    t = label.lower()
    # Normalize label aliases
    if t.startswith("world"):
        return "World"
    if t.startswith("sports") or t.startswith("sport"):
        return "Sports"
    if t.startswith("business") or t.startswith("biz"):
        return "Business"
    if t.startswith("sci/tech") or t.startswith("sci-tech") or t.startswith("sci") or t.startswith("science") or t.startswith("technology") or t.startswith("tech"):
        return "Sci/Tech"

    # Fallback: search the entire text for the first occurrence of the four classes
    t_full = ans.lower()
    idx = []
    for key, norm in [
        ("world", "World"),
        ("sports", "Sports"),
        ("sport", "Sports"),
        ("business", "Business"),
        ("sci/tech", "Sci/Tech"),
        ("sci-tech", "Sci/Tech"),
        ("science", "Sci/Tech"),
        ("technology", "Sci/Tech"),
        ("tech", "Sci/Tech"),
        ("sci", "Sci/Tech"),
    ]:
        pos = t_full.find(key)
        if pos != -1:
            idx.append((pos, norm))
    if idx:
        idx.sort(key=lambda x: x[0])
        return idx[0][1]

    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-jsonl", required=True, help="Model prediction JSONL (each line contains prompt/answer)")
    args = ap.parse_args()

    with open("../../../data/eval/ag_news_input2label.json", 'r', encoding='utf-8') as f:
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
