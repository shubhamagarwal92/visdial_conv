import argparse
import json
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import pickle as pkl
from visdialch.data.dataset import VisDialDataset
from visdialch.encoders import Encoder
from visdialch.decoders import Decoder
from visdialch.metrics import SparseGTMetrics, NDCG, scores_to_ranks
from visdialch.model import EncoderDecoderModel
from visdialch.utils.checkpointing import load_checkpoint


parser = argparse.ArgumentParser(
    "Evaluate and/or generate EvalAI submission file."
)
parser.add_argument(
    "--config-yml",
    default="configs/lf_disc_faster_rcnn_x101.yml",
    help="Path to a config file listing reader, model and optimization "
    "parameters.",
)
parser.add_argument(
    "--split",
    default="val",
    choices=["val", "test"],
    help="Which split to evaluate upon.",
)
parser.add_argument(
    "--val-json",
    default="data/visdial_1.0_val.json",
    help="Path to VisDial v1.0 val data. This argument doesn't work when "
    "--split=test.",
)
parser.add_argument(
    "--val-dense-json",
    default="data/visdial_1.0_val_dense_annotations.json",
    help="Path to VisDial v1.0 val dense annotations (if evaluating on val "
    "split). This argument doesn't work when --split=test.",
)
parser.add_argument(
    "--test-json",
    default="data/visdial_1.0_test.json",
    help="Path to VisDial v1.0 test data. This argument doesn't work when "
    "--split=val.",
)
parser.add_argument(
    "--data_dir",
    default="data/",
    help="Path to data directory.",
)

# SA: if we want to use pre-trained embeddings.
parser.add_argument(
    "--use_pretrained_emb",
    action="store_true",
    help="If we want to use pre-trained embeddings such as BERT.",
)

parser.add_argument(
    "--qa_emb_file_path",
    default="/visdial_1.0_train_emb.h5",
    help="Path to qa embeddings.",
)

parser.add_argument(
    "--hist_emb_file_path",
    default="visdial_1.0_test_emb.h5",
    help="Path to hist embeddings.",
)


parser.add_argument_group("Evaluation related arguments")
parser.add_argument(
    "--load-pthpath",
    default="checkpoints/checkpoint_xx.pth",
    help="Path to .pth file of pretrained checkpoint.",
)
parser.add_argument(
    "--save-dirpath",
    default="checkpoints/",
    help="Path of directory to create checkpoint directory and save "
    "checkpoints.",
)

parser.add_argument(
    "--ignore_caption",
    action="store_false",
    help="If caption should be used as part of history"
)

parser.add_argument_group(
    "Arguments independent of experiment reproducibility"
)
parser.add_argument(
    "--gpu-ids",
    nargs="+",
    type=int,
    default=-1,
    help="List of ids of GPUs to use.",
)
parser.add_argument(
    "--cpu-workers",
    type=int,
    default=4,
    help="Number of CPU workers for reading data.",
)
parser.add_argument(
    "--overfit",
    action="store_true",
    help="Overfit model on 5 examples, meant for debugging.",
)
parser.add_argument(
    "--in-memory",
    action="store_true",
    help="Load the whole dataset and pre-extracted image features in memory. "
    "Use only in presence of large RAM, atleast few tens of GBs.",
)

parser.add_argument_group("Submission related arguments")
parser.add_argument(
    "--save-ranks-path",
    default="logs/ranks.json",
    help="Path (json) to save ranks, in a EvalAI submission format.",
)

# For reproducibility.
# Refer https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# =============================================================================
#   INPUT ARGUMENTS AND CONFIG
# =============================================================================

args = parser.parse_args()

# keys: {"dataset", "model", "solver"}
config = yaml.load(open(args.config_yml))

# SA: Changing relative paths to absolute paths
config["dataset"]["image_features_train_h5"] = "{}/{}".format(args.data_dir, config["dataset"]["image_features_train_h5"])
config["dataset"]["image_features_val_h5"] = "{}/{}".format(args.data_dir, config["dataset"]["image_features_val_h5"])
config["dataset"]["image_features_test_h5"] = "{}/{}".format(args.data_dir, config["dataset"]["image_features_test_h5"])
config["dataset"]["word_counts_json"] = "{}/{}".format(args.data_dir, config["dataset"]["word_counts_json"])
if "glove_npy" in config["dataset"]:
    config["dataset"]["glove_npy"] = "{}/{}".format(args.data_dir, config["dataset"]["glove_npy"])


if isinstance(args.gpu_ids, int):
    args.gpu_ids = [args.gpu_ids]
device = (
    torch.device("cuda", args.gpu_ids[0])
    if args.gpu_ids[0] >= 0
    else torch.device("cpu")
)

print("Running on:", args.gpu_ids)
print("First gpu id", args.gpu_ids[0])
print("Verifying device", device)

# SA: confirm this is working.
# see: https://gist.github.com/shubhamagarwal92/8ecf839cf70c4990e3540d0bb4f288ff
torch.cuda.set_device(device)

# Print config and args.
print(yaml.dump(config, default_flow_style=False))
for arg in vars(args):
    print("{:<20}: {}".format(arg, getattr(args, arg)))

pin_memory = config["solver"].get("pin_memory", True)
print(f"Pin memory is set to {pin_memory}")

# =============================================================================
#   SETUP DATASET, DATALOADER, MODEL
# =============================================================================

if args.split == "val":
    val_dataset = VisDialDataset(
        config["dataset"],
        args.val_json,
        args.val_dense_json,
        use_pretrained_emb=args.use_pretrained_emb,
        overfit=args.overfit,
        in_memory=args.in_memory,
        use_caption=args.use_caption,
        return_options=True,
        add_boundary_toks=False
        if config["model"]["decoder"] != "gen"
        else True,
    )
else:
    val_dataset = VisDialDataset(
        config["dataset"],
        args.test_json,
        use_pretrained_emb=args.use_pretrained_emb,
        overfit=args.overfit,
        in_memory=args.in_memory,
        use_caption=args.ignore_caption,
        return_options=True,
        add_boundary_toks=False
        if config["model"]["decoder"] != "gen"
        else True,
    )
val_dataloader = DataLoader(
    val_dataset,
    batch_size=config["solver"]["batch_size"]
    if config["model"]["decoder"] != "gen"
    else 5,
    num_workers=args.cpu_workers,
    pin_memory=pin_memory
)

# Pass vocabulary to construct Embedding layer.
encoder = Encoder(config["model"], val_dataset.vocabulary)
decoder = Decoder(config["model"], val_dataset.vocabulary)
print("Encoder: {}".format(config["model"]["encoder"]))
print("Decoder: {}".format(config["model"]["decoder"]))

# Share word embedding between encoder and decoder.
decoder.word_embed = encoder.word_embed

# Wrap encoder and decoder in a model.
model = EncoderDecoderModel(encoder, decoder).to(device)
if -1 not in args.gpu_ids:
    model = nn.DataParallel(model, args.gpu_ids)

model_state_dict, _ = load_checkpoint(args.load_pthpath)
if isinstance(model, nn.DataParallel):
    model.module.load_state_dict(model_state_dict)
else:
    model.load_state_dict(model_state_dict)
print("Loaded model from {}".format(args.load_pthpath))

# Declare metric accumulators (won't be used if --split=test)
sparse_metrics = SparseGTMetrics()
ndcg = NDCG()

# =============================================================================
#   EVALUATION LOOP
# =============================================================================

model.eval()
ranks_json = []
# SA: saving log probs for ensembling
opt_log_probs = []
batch_element_list = []

for batch_num, batch in enumerate(tqdm(val_dataloader)):
    for key in batch:
        batch[key] = batch[key].to(device)
    with torch.no_grad():
        output = model(batch)

    # output -> (bs, rounds, options)
    ranks = scores_to_ranks(output)

    # SA: adding previous code here for ensembling
    # if args.split == "test":
        # SA: todo check if we need to append this - No
        # opt_log_probs.append(output.cpu().numpy())
        # SA: confirm if we need to view or already in this shape
    log_softmax=nn.LogSoftmax(dim=-1)
    softmax_probs = log_softmax(output)
    log_probs = output.view(-1, 10, 100).cpu().numpy()
    softmax_probs = softmax_probs.view(-1, 10, 100).cpu().numpy()
    # print(log_probs.shape)

    for i in range(len(batch["img_ids"])):
        # Cast into types explicitly to ensure no errors in schema.
        # Round ids are 1-10, not 0-9
        if args.split == "test":
            ranks_json.append(
                {
                    "image_id": batch["img_ids"][i].item(),
                    "round_id": int(batch["num_rounds"][i].item()),
                    "ranks": [
                        rank.item()
                        for rank in ranks[i][batch["num_rounds"][i] - 1]
                    ],
                }
            )
            # SA: adding previous code here for ensembling
            opt_log_probs.append(list(log_probs[i][batch['num_rounds'][i] - 1]))

        else:
            # SA: note todo all ranks are stored here..and not just for dense annotation
            for j in range(batch["num_rounds"][i]):
                ranks_json.append(
                    {
                        "image_id": batch["img_ids"][i].item(),
                        "round_id": int(j + 1),
                        "ranks": [rank.item() for rank in ranks[i][j]],
                    }
                )
            # num_rounds will be 10 for val..however round_id is used for dense
            # NOTE: careful - round_id -> 1-index
            if "gt_relevance" in batch:
                opt_log_probs.append(
                    {
                        "image_id": batch["img_ids"][i].item(),
                        "round_id": int(batch["round_id"][i].item()),
                        "log_probs": list(log_probs[i][batch['round_id'][i] - 1]),
                        "softmax_probs": list(softmax_probs[i][batch['round_id'][i] - 1])
                    }
                )
                # opt_log_probs.append(list(log_probs[i][batch['round_id'][i] - 1]))
    if args.split == "val":
        # SA: saving batch element here
        batch_element = {
            "ans_ind": batch["ans_ind"].cpu().numpy(),
            "round_id": batch["round_id"].cpu().numpy(),
            "output": output.cpu().numpy()
        }

        sparse_metrics.observe(output, batch["ans_ind"])
        if "gt_relevance" in batch:
            output = output[
                torch.arange(output.size(0)), batch["round_id"] - 1, :
            ]
            ndcg.observe(output, batch["gt_relevance"])

            # SA: saving batch element
            batch_element["output_gt_relevance"] = output.cpu().numpy()
            batch_element["gt_relevance"] = batch["gt_relevance"].cpu().numpy()
            batch_element["img_ids"] = batch["img_ids"].cpu().numpy()

        # SA: saving batch element here
        batch_element_list.append(batch_element)

print("Total batches considered: ", batch_num + 1)

if args.split == "val":
    all_metrics = {}
    all_metrics.update(sparse_metrics.retrieve(reset=True))
    all_metrics.update(ndcg.retrieve(reset=True))
    for metric_name, metric_value in all_metrics.items():
        print(f"{metric_name}: {metric_value}")

print("Writing ranks to {}".format(args.save_ranks_path))
os.makedirs(os.path.dirname(args.save_ranks_path), exist_ok=True)
json.dump(ranks_json, open(args.save_ranks_path, "w"))


# For test we save np array for ensembling,
# for val the whole json as annotations --> for visualization
path_opt_log_probs = os.path.splitext(args.save_ranks_path)[0].replace(
    'ranks_', 'opt_log_probs_') + ".pkl"

if args.split == "test":
    opt_log_probs = np.array(opt_log_probs)
    print(opt_log_probs.shape)

with open(path_opt_log_probs, 'wb') as fp:
    pkl.dump(opt_log_probs, fp)

# TypeError: Object of type 'float32' is not JSON serializable
# if args.split == "val":
#     path_opt_log_probs = os.path.splitext(args.save_ranks_path)[0].replace(
#         'ranks_', 'opt_log_probs_') + ".json"
#     with open(path_opt_log_probs, 'w') as outfile:
#         json.dump(opt_log_probs, outfile)


print(f"Saving the output log probs to {path_opt_log_probs}")

# To get oracle preds
if args.split == "val":
    # SA: save option log probs -> change file path from ranks to opt_log_probs
    path_batch_element = os.path.splitext(args.save_ranks_path)[0].replace(
        'ranks_', 'batch_element_') + ".pkl"
    print(f"Saving the output and batch to {path_batch_element}")
    with open(path_batch_element, 'wb') as fp:
        pkl.dump(batch_element_list, fp)
