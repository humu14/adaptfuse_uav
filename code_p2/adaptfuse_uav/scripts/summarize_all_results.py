import os
import json
import glob

log_dir = r"e:\Research Projects\reserch writing\Co-Sup\code_p2\adaptfuse_uav\outputs\logs"
results = {}
for f in glob.glob(os.path.join(log_dir, "*_test_results.json")):
    name = os.path.basename(f).replace("_test_results.json", "")
    with open(f, "r") as fp:
        results[name] = json.load(fp).get("test", {})

print(f"{'Experiment':38} | {'Disaster F1':11} | {'Victim F1':10} | {'Comb F1':8} | {'Dis Acc':8} | {'Vic Acc':8} | {'Loss':7}")
print("-" * 95)
for k in sorted(results.keys()):
    v = results[k]
    d_f1 = v.get("disaster_f1", 0.0)
    v_f1 = v.get("victim_f1", 0.0)
    c_f1 = v.get("combined_f1", 0.0)
    d_acc = v.get("disaster_acc", 0.0)
    v_acc = v.get("victim_acc", 0.0)
    loss = v.get("loss", 0.0)
    print(f"{k:38} | {d_f1:11.4f} | {v_f1:10.4f} | {c_f1:8.4f} | {d_acc:8.4f} | {v_acc:8.4f} | {loss:7.4f}")

# Also dump to all_results_summary.json
summary_path = os.path.join(log_dir, "all_results_summary.json")
with open(summary_path, "w") as fp:
    json.dump(results, fp, indent=2)
print(f"\nSaved summary to {summary_path}")
