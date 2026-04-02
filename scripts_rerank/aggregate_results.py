import os
import json
import argparse
import numpy as np
from scipy import stats
import collections
import re

DEFAULT_OUTPUT_FILE = "aggregated_results.tsv"

def calculate_ci(data):
    if not data:
        return 0, 0, 0, 0
    n = len(data)
    mean = np.mean(data)
    if n < 2:
        return mean, 0, 0, 0
    std = np.std(data, ddof=1)
    
    # Normal distribution CI (95%)
    z_critical = 1.96
    normal_margin = z_critical * (std / np.sqrt(n))
    
    # Student's t-distribution CI (95%)
    t_critical = stats.t.ppf(0.975, n-1)
    t_margin = t_critical * (std / np.sqrt(n))
    
    return mean, std, normal_margin, t_margin

def extract_timestamp(dir_name):
    match = re.search(r'(\d{8}T\d{6})', dir_name)
    return match.group(1) if match else ""

def read_token_stats(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if data.get("aid") == "all":
                    ts = data.get("token_stats", {})
                    input_tokens = ts.get("input_tokens", 0) or ts.get("prompt_tokens", 0)
                    input_tokens_details = ts.get("input_tokens_details", {})
                    if not input_tokens_details:
                        input_tokens_details = ts.get("prompt_tokens_details", {})
                    cached_tokens = input_tokens_details.get("cached_tokens", 0)
                    output_tokens = ts.get("output_tokens", 0) or ts.get("completion_tokens", 0)
                    output_tokens_details = ts.get("output_tokens_details", {})
                    if not output_tokens_details:
                        output_tokens_details = ts.get("completion_tokens_details", {})
                    reasoning_tokens = output_tokens_details.get("reasoning_tokens", 0)
                    return input_tokens, cached_tokens, output_tokens, reasoning_tokens
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

def find_summary_files(base_dir):
    summaries = collections.defaultdict(lambda: collections.defaultdict(list))
    
    for root, dirs, files in os.walk(base_dir):
        if "evaluation_summary.json" in files:
            parts = root.split(os.sep)
            eval_dir_idx = next((i for i, part in enumerate(parts) if part.startswith("evals_")), None)

            # If this path does not contain an evals_* directory, skip it.
            if eval_dir_idx is None:
                continue

            run_dir = os.sep.join(parts[:eval_dir_idx])
            eval_dir_name = parts[eval_dir_idx]
            
            if "gpt-oss-20b" in eval_dir_name:
                judge_model = "gpt-oss-20b"
            elif "gpt-oss-120b" in eval_dir_name:
                judge_model = "gpt-oss-120b"
            else:
                continue
            
            timestamp = extract_timestamp(eval_dir_name)
            file_path = os.path.join(root, "evaluation_summary.json")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    summaries[run_dir][judge_model].append((timestamp, data))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                    
    for run_dir in summaries:
        for judge_model in summaries[run_dir]:
            summaries[run_dir][judge_model].sort(key=lambda x: x[0])
            if len(summaries[run_dir][judge_model]) > 5:
                summaries[run_dir][judge_model] = summaries[run_dir][judge_model][-5:]
            summaries[run_dir][judge_model] = [x[1] for x in summaries[run_dir][judge_model]]
            
    return summaries

def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate evaluation summaries into TSV.")
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Base directory to search for evaluation_summary.json files.",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help="Output TSV file path. If relative, it is resolved under --base-dir.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.abspath(args.base_dir)
    output_file = args.output_file
    if not os.path.isabs(output_file):
        output_file = os.path.join(base_dir, output_file)

    summaries = find_summary_files(base_dir)
    
    headers = [
        "Run Directory",
        "Recall", "Searches",
        "input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens",
        "ih_input_tokens", "ih_cached_tokens", "ih_output_tokens", "ih_reasoning_tokens",
        "Acc 20b E1", "Acc 20b E2", "Acc 20b E3", "Acc 20b E4", "Acc 20b E5",
        "Acc 20b Avg", "Acc 20b CI Normal", "Acc 20b CI t",
        "Calib 20b E1", "Calib 20b E2", "Calib 20b E3", "Calib 20b E4", "Calib 20b E5",
        "Calib 20b Avg", "Calib 20b CI Normal", "Calib 20b CI t",
        "Acc 120b E1", "Acc 120b E2", "Acc 120b E3", "Acc 120b E4", "Acc 120b E5",
        "Acc 120b Avg", "Acc 120b CI Normal", "Acc 120b CI t",
        "Calib 120b E1", "Calib 120b E2", "Calib 120b E3", "Calib 120b E4", "Calib 120b E5",
        "Calib 120b Avg", "Calib 120b CI Normal", "Calib 120b CI t"
    ]
    
    with open(output_file, 'w') as f:
        f.write("\t".join(headers) + "\n")
        
        for run_dir in sorted(summaries.keys()):
            # Token stats
            token_stats_file = os.path.join(run_dir, "aggregated_token_stats.jsonl")
            ih_token_stats_file = os.path.join(run_dir, "invocation_history", "aggregated_token_stats.jsonl")
            
            main_tokens = read_token_stats(token_stats_file)
            ih_tokens = read_token_stats(ih_token_stats_file)
            
            if main_tokens is None: main_tokens = (0, 0, 0, 0)
            if ih_tokens is None: ih_tokens = (0, 0, 0, 0)
            
            # Model data collection
            m_data = {}
            for model in ["gpt-oss-20b", "gpt-oss-120b"]:
                evals = summaries[run_dir].get(model, [])
                accuracies = [e.get("Accuracy (%)", 0) for e in evals]
                calibrations = [e.get("Calibration Error (%)", 0) for e in evals]
                recalls = [e.get("Recall (%)", 0) for e in evals]
                searches = [e.get("avg_tool_stats", {}).get("search", 0) for e in evals]
                
                m_data[model] = {
                    'recall': recalls[0] if recalls else 0,
                    'search': searches[0] if searches else 0,
                    'accs': (accuracies + [None]*5)[:5],
                    'acc_stats': calculate_ci(accuracies),
                    'calibs': (calibrations + [None]*5)[:5],
                    'calib_stats': calculate_ci(calibrations)
                }
            
            recall = m_data['gpt-oss-120b']['recall'] if m_data['gpt-oss-120b']['recall'] else m_data['gpt-oss-20b']['recall']
            search = m_data['gpt-oss-120b']['search'] if m_data['gpt-oss-120b']['search'] else m_data['gpt-oss-20b']['search']
            
            row = [run_dir.replace(base_dir + "/", "")]
            row.append(f"{recall:.2f}")
            row.append(f"{search:.2f}")
            
            # Add token stats
            row.extend([str(t) for t in main_tokens])
            row.extend([str(t) for t in ih_tokens])
            
            # 20b Acc
            row.extend([f"{a:.2f}" if a is not None else "N/A" for a in m_data['gpt-oss-20b']['accs']])
            row.extend([f"{m_data['gpt-oss-20b']['acc_stats'][0]:.2f}", f"{m_data['gpt-oss-20b']['acc_stats'][2]:.2f}", f"{m_data['gpt-oss-20b']['acc_stats'][3]:.2f}"])
            
            # 20b Calib
            row.extend([f"{c:.2f}" if c is not None else "N/A" for c in m_data['gpt-oss-20b']['calibs']])
            row.extend([f"{m_data['gpt-oss-20b']['calib_stats'][0]:.2f}", f"{m_data['gpt-oss-20b']['calib_stats'][2]:.2f}", f"{m_data['gpt-oss-20b']['calib_stats'][3]:.2f}"])
            
            # 120b Acc
            row.extend([f"{a:.2f}" if a is not None else "N/A" for a in m_data['gpt-oss-120b']['accs']])
            row.extend([f"{m_data['gpt-oss-120b']['acc_stats'][0]:.2f}", f"{m_data['gpt-oss-120b']['acc_stats'][2]:.2f}", f"{m_data['gpt-oss-120b']['acc_stats'][3]:.2f}"])
            
            # 120b Calib
            row.extend([f"{c:.2f}" if c is not None else "N/A" for c in m_data['gpt-oss-120b']['calibs']])
            row.extend([f"{m_data['gpt-oss-120b']['calib_stats'][0]:.2f}", f"{m_data['gpt-oss-120b']['calib_stats'][2]:.2f}", f"{m_data['gpt-oss-120b']['calib_stats'][3]:.2f}"])
            
            f.write("\t".join(row) + "\n")

    print(f"Aggregated results saved to {output_file}")

if __name__ == "__main__":
    main()
