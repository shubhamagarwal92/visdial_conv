## visdial_conv [![twitter](https://img.shields.io/twitter/url/https/shields.io.svg?style=social)](https://ctt.ac/rlwuX)

[![Paper](https://img.shields.io/badge/Presented%20at%20-ACL2020-yellow)](https://acl2020.org/program/accepted/)
[![CodeAMT](https://img.shields.io/badge/code-AMT%20interface-green.svg)](https://github.com/shubhamagarwal92/visdialconv-amt)

This repository contains code used in our ACL'20 paper [History for Visual Dialog: Do we really need it?](https://arxiv.org/pdf/2005.07493.pdf)

## Credits

This repository is build upon [visdial-challenge-starter-pytorch](https://github.com/batra-mlp-lab/visdial-challenge-starter-pytorch). Previous commit history is maintained. We thank the challenge organizers for providing the starter code.  
Plese see `original_README.md` or point to the original repo to setup the conda environment and download the relevant data.

Alternatively, we provide [setup.sh](/setup_visdial/setup.sh) to streamline the process. Run as 

```
cd setup_visdial
bash setup.sh
```

We follow the directory structure

```
$PROJECT_DIR
    |--$DATA_DIR==data 
    |--$MODEL_DIR==models
    |--$CODE_DIR==visdial_conv
        |--$CONFIG_DIR==configs
```

We used Python3 for our experiments and PyTorch 1.0.0/1.0.1post2 for our experiments. We oftenly use `f-strings` and `typing` in our code. Some basic familiarity is required. 

Installation using docker can be found [here](./docker).

## Code

Update: v0.1 of the code has been released. We suggest to use PyCharm for this project. See [this blog](https://medium.com/analytics-vidhya/code-like-a-pro-ish-right-from-101-tools-from-a-deep-learning-perspective-34d8df1e38e#42e8) to get more details. 

We provide [shell scripts](./shell_scripts) to run our models. To reproduce the results for different models, follow these scripts:

- [MCA-I](./shell_scripts/train_and_evaluate_mcan_img_only.sh)
- [MCA-I-H](./shell_scripts/train_and_evaluate_mcan_img_mcan_hist.sh)
- [MCA-VGH-I](./shell_scripts/train_and_evaluate_mcan_img_mcan_vqa_hist_attn.sh)
- [MCA-I-HGuidedQ](./shell_scripts/train_and_evaluate_hist_guided_qmcan.sh)
- [MCA-I-H-GT](./new_annotations/train_and_evaluate_mcan_img_mcan_hist.sh)
( The Python script in the same [new_annotations](./new_annotations) folder shows how we fixed dense gt annotations.)


We follow the same directory structure as described above in all the shell scripts.   

Some jupyter notebooks for inspection of data/images/analyses/results can be found in [notebooks](./notebooks).
Run `conda install jupyter` if you are using conda and want to run these notebooks from the environment. More data analysis is provided in [data_analysis](./data_analysis) folder.

We have also provided some test cases in the [tests](./tests) folder. We strongly suggest to add to this folder and test your new python scripts if you build on top of this repository. 

Our code follows this structure: 

- [train.py](train.py) -- entrypoint for training. Called by all shell scripts
- [evaluate.py](evaluate.py) -- python script for evaluation. Called by all shell scripts
- [data](./visdialch/data) -- dataset reader and vocabulary defined here
- [encoders](./visdialch/encoders) -- all encoders defined here
- [decoders](./visdialch/decoders) -- all decoders to be defined here
- [model.py](./visdialch/model.py) -- wrapper to call models with different encoders and decoders
- [metrics.py](./visdialch/metrics.py) -- define NDCG and other metrics
- [configs](./configs) -- all configs defined here
- [shell_scripts](./shell_scripts) -- all shell scripts here

Be careful about different indexing in the data. See [notes.txt](./notes.txt)

## Dataset

We have released two subsets of Visdial val set (mentioned in our paper) in the folder [released_datasets](./released_datasets): 
1. VisdialConv - Instances which require dialog history verified by crowdsourced human annotations
2. Vispro - Intersection of Vispro and Visdial val set

To evaluate on these subsets, use the shell scripts provided in [evaluate_subset_data](./evaluate_subset_data).

We used the scripts in [subset_dialog_data](./subset_dialog_data) to create these subsets from VisdialVal set.

If you are interested in our AMT interface, please refer to the [repository](https://github.com/shubhamagarwal92/visdialconv-amt).

See the [README](./released_datasets/visdialconv/README.md) in the visdialconv folder to know more about the annotations. 

## Citation

If you use this work, please cite it as
```
@inproceedings{agarwal2020history,
  title={History for Visual Dialog: Do we really need it?},
  author={Agarwal, Shubham and Bui, Trung and Lee, Joon-Young and Konstas, Ioannis and Rieser, Verena},
  booktitle={58th Annual meeting of the Association for Computational Linguistics (ACL)},
  year={2020}
}
```

Feel free to fork and contribute to this work. Please raise a PR or any related issues. Will be happy to help. Thanks.

Badges made using [shields.io](https://shields.io/)
