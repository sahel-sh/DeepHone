import os
import json
import csv
import argparse
from pathlib import Path

def aggregate_token_stats(root_dir, output_csv):
    results = []
    root_path = Path(root_dir)
    
    # Walk through the directory to find all aggregated_token_stats.jsonl files
    # specifically under 'invocation_history' folders
    for path in root_path.rglob('invocation_history/aggregated_token_stats.jsonl'):
        file_path = str(path.absolute())
        print(f"Processing {file_path}")
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            if not lines:
                continue
                
            valid_queries_count = 0
            total_reasoning_tokens = 0
            aggregate_found = False
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # Check if it's the aggregate line
                if data.get('qid') == 'all' or data.get('aid') == 'all':
                    aggregate_found = True
                    token_stats = data.get('token_stats', {})
                    output_details = token_stats.get('output_tokens_details', {})
                    if not output_details:
                        output_details = token_stats.get('completion_tokens_details', {})

                    total_reasoning_tokens = output_details.get('reasoning_tokens', 0)
                else:
                    # It's a query line. Check if it has valid token usage stats.
                    token_stats = data.get('token_stats', {})
                    # A valid query must have token stats (not empty)
                    if token_stats and 'total_tokens' in token_stats:
                        valid_queries_count += 1
            
            if aggregate_found:
                results.append({
                    'file_path': file_path,
                    'num_queries': valid_queries_count,
                    'total_reasoning_tokens': total_reasoning_tokens
                })
            else:
                print(f"Warning: No aggregate line found in {file_path}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Write results to CSV
    if not os.path.isabs(output_csv):
        output_csv = os.path.join(root_dir, output_csv)
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['file_path', 'num_queries', 'total_reasoning_tokens']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in results:
            writer.writerow(row)
            
    print(f"Aggregation complete. Results saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate reasoning token stats across run directories."
    )
    parser.add_argument(
        "--runs_dir",
        required=True,
        help="Root runs directory to scan for invocation_history/aggregated_token_stats.jsonl.",
    )
    parser.add_argument(
        "--output_filename",
        default="reasoning_token_stats_summary.csv",
        help="Output CSV file path.",
    )
    args = parser.parse_args()

    aggregate_token_stats(args.runs_dir, args.output_filename)

