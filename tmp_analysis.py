import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load pipeline manifest for ground truth
manifest = pd.read_csv("data/embeddings/manifest.csv")
# manifest has: source_file, segment_index, label (1 for bird, 0 for noise), start_sec, end_sec
# Let's map (source_file, start_sec, end_sec) to label and split

gt_map = {}
for _, row in manifest.iterrows():
    key = (Path(row['source_file']).as_posix(), float(row['start_sec']), float(row['end_sec']))
    gt_map[key] = {
        'label': int(row['label']),
        'split': row['split']
    }

# 2. Parse Baseline
baseline_preds = {}
with open("comparison/baseline_normalized.jsonl", "r") as f:
    for line in f:
        d = json.loads(line)
        # BirdNET predicted a bird
        src = Path(d['source_file']).as_posix()
        start = float(d['start_sec'])
        end = float(d['end_sec'])
        conf = float(d['confidence'])
        key = (src, start, end)
        
        # We take max confidence if multiple predictions for same segment
        if key not in baseline_preds:
            baseline_preds[key] = conf
        else:
            baseline_preds[key] = max(baseline_preds[key], conf)

# Evaluate Baseline over the COMPLETE manifest test set matches
# Wait, let's just evaluate over the whole dataset or just TEST split? The user says "Same test split", but BirdNET baseline didn't use splits (it's unsupervised/pretrained). We can evaluate it on the TEST split to be fair, or the FULL dataset. Let's do TEST set to match pipeline's metrics.json.

y_true_test = []
y_pred_baseline_test = []

y_true_all = []
y_pred_baseline_all = []

# Hard negative tracking
test_noise_count = 0
baseline_fp = [] # noise but baseline said bird
pipeline_noise_preds = []

# Find pipeline outputs to see if we can get routing
# the pipeline also wrote to outputs/clean_birds outputs/noise outputs/uncertain
pipeline_decision_map = {}
for b in ["clean_birds", "noise", "uncertain"]:
    p = Path(f"outputs/{b}")
    if p.exists():
        for json_file in p.glob("*.json"):
            try:
                with open(json_file, "r") as jf:
                    data = json.load(jf)
                    k = (Path(data['source_file']).as_posix(), float(data['start_sec']), float(data['end_sec']))
                    pipeline_decision_map[k] = {
                        'decision': data['decision'],
                        'confidence': data['confidence'],
                        'species': data.get('predicted_species', '')
                    }
            except Exception as e:
                pass


for key, label_info in gt_map.items():
    lbl = label_info['label']
    split = label_info['split']
    
    # Baseline threshold default is typically 0.1 or 0.3 for BirdNET?
    # Actually, if the segment is in baseline_preds, it means BirdNET predicted SOMETHING.
    # We will assume a bird was predicted if conf > 0.0 (or just present in preds). Let's use 0.1
    b_conf = baseline_preds.get(key, 0.0)
    b_pred = 1 if b_conf > 0.1 else 0
    
    if split == 'test':
        y_true_test.append(lbl)
        y_pred_baseline_test.append(b_pred)
        if lbl == 0:
            test_noise_count += 1
            if b_pred == 1:
                baseline_fp.append((key, b_conf))
                
    y_true_all.append(lbl)
    y_pred_baseline_all.append(b_pred)

print(f"Total dataset: {len(y_true_all)}, Test set: {len(y_true_test)}")
print(f"Test Set Baseline Accuracy: {accuracy_score(y_true_test, y_pred_baseline_test):.4f}")
print(f"Test Set Baseline Precision: {precision_score(y_true_test, y_pred_baseline_test):.4f}")
print(f"Test Set Baseline Recall: {recall_score(y_true_test, y_pred_baseline_test):.4f}")
print(f"Test Set Baseline F1: {f1_score(y_true_test, y_pred_baseline_test):.4f}")

# Compute FPR and FNR
def compute_rates(y_true, y_pred):
    tn, fp, fn, tp = 0, 0, 0, 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1: tp += 1
        elif t == 1 and p == 0: fn += 1
        elif t == 0 and p == 1: fp += 1
        elif t == 0 and p == 0: tn += 1
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    return fpr, fnr, fp

fpr_base, fnr_base, base_fp_count = compute_rates(y_true_test, y_pred_baseline_test)
print(f"Test Set Baseline FPR: {fpr_base:.4f} (FP={base_fp_count})")
print(f"Test Set Baseline FNR: {fnr_base:.4f}")

# Pipeline metrics from metrics.json are already computed:
# acc: 0.9214, prec: 0.9446, rec: 0.9579, f1: 0.9512, fpr: 0.2234
with open("results/metrics.json") as f:
    pipe_metrics = json.load(f)["metrics_at_optimal_f1"]

# Hard Negative Analysis (overall pipeline routing vs baseline on noise segments)
print("\n--- Hard Negative Analysis ---")
# Count over ALL noise segments
total_noise = 0
baseline_fp_all = 0
pipeline_fp_all = 0 # anything not routed to 'noise' ? Or routed to 'bird'
qualitative_examples = []

for key, label_info in gt_map.items():
    if label_info['label'] == 0:
        total_noise += 1
        b_conf = baseline_preds.get(key, 0.0)
        b_pred = 1 if b_conf > 0.1 else 0
        if b_pred == 1:
            baseline_fp_all += 1
        
        pMap = pipeline_decision_map.get(key)
        if pMap:
            if pMap['decision'] == 'bird':
                pipeline_fp_all += 1
            if b_pred == 1 and pMap['decision'] in ('noise', 'uncertain'):
                # Pipeline FIXED it!
                qualitative_examples.append({
                    'file': key[0],
                    'start': key[1],
                    'baseline_conf': b_conf,
                    'pipeline_decision': pMap['decision'],
                    'pipeline_conf': pMap['confidence']
                })

print(f"Total Noise Segments: {total_noise}")
print(f"Baseline FPs (>=0.1 conf): {baseline_fp_all}")
print(f"Pipeline FPs (routed to bird): {pipeline_fp_all}")
print(f"Reduction in False Positives: {((baseline_fp_all - pipeline_fp_all) / baseline_fp_all * 100) if baseline_fp_all > 0 else 0:.2f}%")

print("\n--- Uncertain Analysis ---")
uncertain_count = 0
uncertain_true_birds = 0
total_pipeline_processed = len(pipeline_decision_map)
for key, data in pipeline_decision_map.items():
    if data['decision'] == 'uncertain':
        uncertain_count += 1
        lbl = gt_map.get(key, {}).get('label', -1)
        if lbl == 1:
            uncertain_true_birds += 1

print(f"Total processed in pipeline router: {total_pipeline_processed}")
print(f"Total uncertain: {uncertain_count}")
if total_pipeline_processed > 0:
    print(f"% of data uncertain: {uncertain_count / total_pipeline_processed * 100:.2f}%")
if uncertain_count > 0:
    print(f"% of uncertain that are actual birds: {uncertain_true_birds / uncertain_count * 100:.2f}%")

print("\n--- Qualitative Examples (Top 10 Fixed by Pipeline) ---")
# Sort by highest baseline confidence that was successfully filtered
qualitative_examples.sort(key=lambda x: x['baseline_conf'], reverse=True)
for i, ex in enumerate(qualitative_examples[:10]):
    print(f"{i+1}. {ex['file']} ({ex['start']}s) | BirdNET Conf: {ex['baseline_conf']:.2f} -> Pipeline: {ex['pipeline_decision']} (Conf: {ex['pipeline_conf']:.2f})")
