

for d in ./results/*/; do python3 code_results_summary.py "$d" --no-plot; done
python3 tool_concatenate_ranking_results.py



