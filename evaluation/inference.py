import csv
import json
import time
import os
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import sys

# Fixed configuration parameters
retry_delay = 30
max_workers = 50

def send_request_to_model(prompt, client, model, max_tokens=1024, max_retries=3):
    """Send a request to the model and get a response"""
    for attempt in range(max_retries):
        try:
            print(f"[DEBUG] Processing request, attempt {attempt + 1}/{max_retries}")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,  # Use 2048 for LRM models, otherwise 1024
                temperature=0,
            )
            print(f"[DEBUG] Request succeeded")
            # Although harmfulness evaluation only uses the answer (not the CoT),
            # for observation purposes, return the full output without truncation.
            return response.choices[0].message.content.strip() 
        except Exception as e:
            print(f"[ERROR] Request failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                sleep_time = min(retry_delay, 2 ** attempt)  # Exponential backoff, capped at retry_delay
                print(f"[INFO] Waiting {sleep_time} seconds before retrying...")
                time.sleep(sleep_time)
    
    return "[ERROR] All retries failed"

def process_single_prompt(args):
    """Process a single prompt - modified to accept tuple arguments for ThreadPoolExecutor"""
    prompt, index, client, model, max_tokens = args
    try:
        print(f"[INFO] Starting prompt #{index}")
        response = send_request_to_model(prompt, client, model, max_tokens)
        print(f"[SUCCESS] Completed prompt #{index}")
        return {
            "index": index,
            "prompt": prompt,
            "answer": response,
            "success": True
        }
    except Exception as e:
        print(f"[ERROR] Failed to process prompt #{index}: {str(e)}")
        return {
            "index": index,
            "prompt": prompt,
            "answer": f"Processing failed: {str(e)}",
            "success": False
        }

def process_csv_file(input_csv_file, api_key, api_base, output_file=None, max_workers_override=None, max_tokens=1024):
    """Process a CSV file"""
    print(f"[INFO] Starting CSV processing: {input_csv_file}")
    
    # Use the provided max_workers or the global default
    workers = max_workers_override if max_workers_override else max_workers
    print(f"[INFO] Using {workers} concurrent threads")
    
    # Initialize client
    print(f"[INFO] Initializing OpenAI client, API Base: {api_base}")
    client = OpenAI(
        api_key=api_key,
        base_url=api_base,
    )
    
    # Get model info
    try:
        print("[INFO] Fetching model list...")
        models = client.models.list()
        print(f"[DEBUG] Available models: {models}")
        model = models.data[0].id
        print(f"[INFO] Using model: {model}")
    except Exception as e:
        print(f"[ERROR] Failed to retrieve model information: {e}")
        return
    
    # Determine output file path
    if output_file is None:
        # Default output path
        input_filename = os.path.splitext(os.path.basename(input_csv_file))[0]
        output_dir = "results"
        output_file = os.path.join(output_dir, f"{input_filename}_responses.jsonl")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        print(f"[INFO] Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    # Read CSV file
    prompts = []
    try:
        print(f"[INFO] Reading CSV file: {input_csv_file}")
        with open(input_csv_file, 'r', encoding='utf-8') as csvfile:
            # Assume CSV has one column, each line is a prompt
            # If it has headers, adjust delimiter and parameters
            reader = csv.reader(csvfile)
            for row_idx, row in enumerate(reader):
                if row and len(row) > 0:  # Ensure row is not empty
                    prompt = row[0].strip()  # Take first column
                    if prompt:  # Ensure prompt is not empty
                        prompts.append((prompt, row_idx))
                        if row_idx < 3:  # Print first 3 prompts for debugging
                            print(f"[DEBUG] Prompt {row_idx}: {prompt[:100]}...")
    except FileNotFoundError:
        print(f"[ERROR] Input file not found: {input_csv_file}")
        return
    except Exception as e:
        print(f"[ERROR] Error reading CSV file: {e}")
        return
    
    print(f"[INFO] Loaded {len(prompts)} prompts from {input_csv_file}")
    
    if len(prompts) == 0:
        print("[ERROR] No valid prompts found")
        return
    
    # Prepare task parameters
    task_args = [(prompt, idx, client, model, max_tokens) for prompt, idx in prompts]
    
    # Process in parallel using thread pool
    print(f"[INFO] Starting parallel processing with {workers} threads")
    results = []
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        print("[INFO] Submitting all tasks to thread pool")
        futures = {executor.submit(process_single_prompt, args): args for args in task_args}
        
        # Open output file
        with open(output_file, 'w', encoding='utf-8') as fout:
            # Collect results
            completed_count = 0
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing progress"):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Write results to JSONL file (omit extra fields)
                    output_record = {
                        "prompt": result["prompt"],
                        "answer": result["answer"]
                    }
                    fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                    fout.flush()  # Ensure timely write
                    
                    completed_count += 1
                    if completed_count % 10 == 0:  # Print progress every 10 items
                        print(f"[INFO] Completed {completed_count}/{len(futures)} tasks")
                        
                except Exception as e:
                    print(f"[ERROR] Error while retrieving task result: {e}")
    
    # Summarize results
    success_count = sum(1 for r in results if r.get("success", False))
    print(f"[INFO] Processing complete: {success_count}/{len(prompts)} tasks succeeded")
    print(f"[INFO] Results saved to: {output_file}")
    return output_file

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Process text prompts in a CSV file and query API for responses')
    
    parser.add_argument('csv_file', 
                       help='Path to the input CSV file')
    
    parser.add_argument('--api-key', 
                       required=True,
                       help='OpenAI API key')
    
    parser.add_argument('--api-base', 
                       required=True,
                       help='Base URL of the API')
    
    parser.add_argument('--output', '-o',
                       help='Output file path (default: results/{input_filename}_responses.jsonl)')
    
    parser.add_argument('--max-workers',
                       type=int,
                       default=50,
                       help='Maximum number of concurrent threads (default: 50)')
    
    parser.add_argument('--debug', 
                       action='store_true',
                       help='Enable debug mode for verbose logging')
    
    parser.add_argument('--max-tokens',
                   type=int,
                   default=1024,
                   help='Maximum number of tokens for model generation (default: 1024)')
    
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Enable debug mode
    if args.debug:
        print("[DEBUG] Debug mode enabled")
    
    # Check if input file exists
    if not os.path.exists(args.csv_file):
        print(f"[ERROR] Input file not found: {args.csv_file}")
        return
    
    print("=" * 50)
    print("Starting CSV processing...")
    print(f"Input file: {args.csv_file}")
    print(f"API Base: {args.api_base}")
    print(f"Concurrency: {args.max_workers}")
    if args.output:
        print(f"Output file: {args.output}")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        output_path = process_csv_file(
            args.csv_file, 
            args.api_key, 
            args.api_base, 
            args.output,
            args.max_workers,
            args.max_tokens
        )
        
        end_time = time.time()
        print("=" * 50)
        print(f"✅ Processing complete!")
        print(f"📊 Total time: {end_time - start_time:.2f} seconds")
        print(f"📁 Output file: {output_path}")
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\n[INFO] Program interrupted by user")
    except Exception as e:
        print(f"[ERROR] An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
