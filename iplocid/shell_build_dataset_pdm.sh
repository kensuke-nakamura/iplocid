python3 tool_build_dataset_json_pdm.py \
  --original_iploc_json ./data/iploc/2_shots_pdm.json \
  --burst_annotations_json /ssd1/dataset/ICL_tracking/video/burst/annotations/test/all_classes.json \
  --burst_frames_base_dir /ssd1/dataset/ICL_tracking/video/burst/frames \
  --out_dir ./data \
  --out_prefix pdm \
  --export_root "/ssd1/dataset/ICL_tracking_minimized/video/burst/frames"
