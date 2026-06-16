
#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="/ssd1/dataset/ICL_tracking"
DST_ROOT="/ssd1/dataset/ICL_tracking_minimized"
DATA_DIR="./data"

SCRIPT="./tool_extract_dataset_from_json.py"

for JSON_PATH in "${DATA_DIR}"/*.json; do
    if [ ! -f "${JSON_PATH}" ]; then
        echo "[WARNING] No JSON files found in ${DATA_DIR}"
        exit 0
    fi

    echo "============================================================"
    echo "[INFO] Processing: ${JSON_PATH}"
    echo "============================================================"

    python3 "${SCRIPT}" \
        "${JSON_PATH}" \
        "${DST_ROOT}" \
        --src_root "${SRC_ROOT}" \
        --write_json
done

echo "============================================================"
echo "[INFO] All done."
echo "============================================================"


