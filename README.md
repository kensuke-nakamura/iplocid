




<p align="center">
  <img src="overview.png">
</p>

<p align="center">
  <em>Figure 1. Framework of IPLoc-ID.</em>
</p>



# The official code of IPLoc-ID

This repository provides the HuggingFace implementation (Pytorch) of the paper:

** Personalized Object Identification and Localization via In-Context Inference of Vision-Language Models **

<!--
Currently, we provide a minimal inference implementation and a trained model for reproducing the inference procedure.

The complete dataset construction scripts, training code, and trained models will be released in accordance with the journal's policy after the paper is published.
-->




# Installation

Create and activate the Conda environment:

```bash
conda create -n iplocid python=3.9 -y
conda activate iplocid
```

Install the required packages:

```bash
pip install -U accelerate pillow
pip install -U git+https://github.com/huggingface/transformers
python -m pip install -U peft
python -m pip install -U torchvision
python -m pip install -U pandas
python -m pip install -U scipy
python -m pip install -U qwen-vl-utils
pip install matplotlib scikit-image
pip install wonderwords
conda install -c conda-forge pycocotools
pip install einops timm
```

Configure Accelerate and select `bf16` when prompted:

```bash
accelerate config
```

# Dataset Preparation

Please prepare the input data according to the dataset format described below.

Step 1: Download the original datasets: LaSOT, PDM (BURST), GOT-10k, and VastTrack, from the official websites.

Step 2: Place the source datasets under the following directory structure:

```text
/ssd1/dataset/ICL_tracking
└── video
    ├── LASOT
    │   └── <class>
    │       └── <subclass>
    ├── burst
    │   ├── annotations
    │   └── frames
    ├── got10k
    │   └── val
    │       └── <class>
    └── VastTrack
        └── <class>
            └── <subclass>
```

Step 3: Run the following command to export the minimum set of images required to run the input data JSON files (`./data/*.json`) from `/ssd1/dataset/ICL_tracking` to `/ssd1/dataset/ICL_tracking_minimized`.

```bash
bash extract_dataset.sh
```

Alternatively, you can generate the input data JSON files from scratch by running the following command.
In this case, ICL_tracking_minimized will also be generated automatically.

```bash
bash shell_build_data-json.sh
```
<!--
-->




# Model Download

Step 1: Download the trained models from the following links:

- [Qwen3-VL-8B-Instruct_iplocid](https://drive.google.com/drive/folders/dummy_link)
- Qwen3-VL-32B-Instruct_iplocid
- Qwen2-VL-7-Instruct_iplocid
  
<!--
- [Qwen3-VL-8B-Instruct_iplocid](https://drive.google.com/drive/folders/dummy_link)
- [Qwen3-VL-32B-Instruct_iplocid](https://drive.google.com/drive/folders/dummy_link)
- [Qwen2-VL-7-Instruct_iplocid](https://drive.google.com/drive/folders/dummy_link)
-->

We also provide our reproduced IPLoc models from the previous method:

- [Qwen3-VL-8B-Instruct_iploc](https://drive.google.com/drive/folders/dummy_link)
- Qwen3-VL-32B-Instruct_iploc
- Qwen2-VL-7-Instruct_iploc
  
<!--
- [Qwen3-VL-8B-Instruct_iploc](https://drive.google.com/drive/folders/dummy_link)
- [Qwen3-VL-32B-Instruct_iploc](https://drive.google.com/drive/folders/dummy_link)
- [Qwen2-VL-7-Instruct_iploc](https://drive.google.com/drive/folders/dummy_link)
-->

If necessary, download the pretrained weights from the previous work, IPLoc, from the **Model Download** section of the following repository:

https://github.com/SivanDoveh/IPLoc

Specifically, please download `QWEN2-VL-ICL-LOC`.

Step 2: Place the pretrained weights as follows:

```text
├── iploc
└── pretrained_weights
    ├── Qwen3-VL-8B-Instruct_iplocid       # our trained IPLoc-ID model
    ├── Qwen3-VL-8B-Instruct_iploc         # our reproduced IPLoc model
    ├── :
    └── Qwen2VL-7b-ICL-Loc                 # original IPLoc model
```

# Inference

Run the inference script as follows:

```bash
bash inference.sh
```

# Training

Run the training script as follows:

```bash
bash training.sh
```
<!--
The complete training code and configuration files will be released after the paper is published.
-->

# Evaluation

For full evaluation of the trained model, please refer to the following script:

```bash
bash evaluation.sh
```



# Citation

The arXiv citation information will be provided here.

```bibtex
@article{iplocid,
  title   = {[Paper Title]},
  author  = {[Author Names]},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

# Acknowledgement

This work is built upon the pioneering work of Doveh et al.:

Sivan Doveh et al.,
“Teaching VLMs to Localize Specific Objects from In-Context Examples,”
Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

```
