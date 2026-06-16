#!/usr/bin/env bash
# shell_build_dataset_LASOT.sh
#
# Example:
#   bash shell_build_dataset_LASOT.sh "1,2,4" "1,2,3,18" 0.2

set -e

N_SET="${1:-1,2,4,8}"
T_SET="${2:-1,2,3,9,18}"
TEST_SAMPLE_RATIO="${3:-0.2}"

LASOT_ROOT="/ssd1/dataset/ICL_tracking/video/LASOT"
EXPORT_ROOT="/ssd1/dataset/ICL_tracking_minimized/video/LASOT"
OUT_DIR="./data"

echo "N_set             : ${N_SET}"
echo "T_set             : ${T_SET}"
echo "test_sample_ratio : ${TEST_SAMPLE_RATIO}"
echo "lasot_root        : ${LASOT_ROOT}"
echo "output_dir        : ${OUT_DIR}"
echo "export_root       : ${EXPORT_ROOT}"

echo ""
echo "------------------------------------------------------------"
echo "[MANUAL / AT YOUR OWN RISK]"
echo "If you want a clean export_root, you may delete it manually before running:"
echo "  rm -rf \"${EXPORT_ROOT}\""
echo "Then recreate the directory if needed:"
echo "  mkdir -p \"${EXPORT_ROOT}\""
echo "------------------------------------------------------------"
echo ""

python3 tool_build_dataset_json_lasot.py \
  --lasot_root "${LASOT_ROOT}" \
  --output_dir "${OUT_DIR}" \
  --export_root "${EXPORT_ROOT}" \
  --test_sample_ratio "${TEST_SAMPLE_RATIO}" \
  --N_set "${N_SET}" \
  --T_set "${T_SET}"

N_LIST=$(echo "${N_SET}" | tr ',' ' ')
T_LIST=$(echo "${T_SET}" | tr ',' ' ')

for N in ${N_LIST}; do
  for T in ${T_LIST}; do
    TEST_JSON="${OUT_DIR}/LASOT_${N}shot_T${T}_classwise-split_test.json"
    TRAIN_JSON="${OUT_DIR}/LASOT_${N}shot_T${T}_classwise-split_train.json"

    if [ "${T}" != "18" ]; then
      if [ -f "${TRAIN_JSON}" ]; then
        echo "Inspecting ${TRAIN_JSON}"
        python3 tool_inspect_dataset_json.py --json_path "${TRAIN_JSON}"
      else
        echo "WARNING: ${TRAIN_JSON} not found."
      fi
    fi

    if [ -f "${TEST_JSON}" ]; then
      echo "Inspecting ${TEST_JSON}"
      python3 tool_inspect_dataset_json.py --json_path "${TEST_JSON}"
    else
      echo "WARNING: ${TEST_JSON} not found."
    fi
  done
done
