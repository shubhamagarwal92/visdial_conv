## visdial_conv

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


## Code

Update: v0.1 of the code has been released. We suggest to use PyCharm for this project. See [this blog](https://medium.com/analytics-vidhya/code-like-a-pro-ish-right-from-101-tools-from-a-deep-learning-perspective-34d8df1e38e#42e8) to get more details. 

We provide [shell scripts](./shell_scripts) to run our models. To reproduce the results for different models, follow these scripts:

- [MCA-I](./shell_scripts/train_and_evaluate_mcan_img_only.sh)
- [MCA-I-H](./shell_scripts/train_and_evaluate_mcan_img_mcan_hist.sh)
- [MCA-VGH-I](./shell_scripts/train_and_evaluate_mcan_img_mcan_vqa_hist_attn.sh)
- [MCA-I-HGuidedQ](./shell_scripts/train_and_evaluate_hist_guided_qmcan.sh)

We follow the same directory structure as described above in all the shell scripts.   

## Dataset

We have released two subsets of Visdial val set (mentioned in our paper) in the folder `released_datasets`: 
1. VisdialConv - Instances which require dialog history verified by crowdsourced human annotations
2. Vispro - Subset of Vispro and Visdial val set

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


