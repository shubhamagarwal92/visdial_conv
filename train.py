import argparse
import itertools

from tensorboardX import SummaryWriter
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from bisect import bisect

from visdialch.data.dataset import VisDialDataset
from visdialch.encoders import Encoder
from visdialch.decoders import Decoder
from visdialch.metrics import SparseGTMetrics, NDCG
from visdialch.model import EncoderDecoderModel
from visdialch.utils.checkpointing import CheckpointManager, load_checkpoint
from visdialch.utils.emb_file_paths import (
    get_emb_dir_file_path,
    get_qa_embeddings_file_path,
    get_hist_embeddings_file_path
)
import datetime
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-yml",
        default="configs/lf_disc_faster_rcnn_x101.yml",
        help="Path to a config file listing reader, model and solver parameters.",
    )
    parser.add_argument(
        "--train-json",
        default="data/visdial_1.0_train.json",
        help="Path to json file containing VisDial v1.0 training data.",
    )
    parser.add_argument(
        "--val-json",
        default="data/visdial_1.0_val.json",
        help="Path to json file containing VisDial v1.0 validation data.",
    )
    parser.add_argument(
        "--val-dense-json",
        default="data/visdial_1.0_val_dense_annotations.json",
        help="Path to json file containing VisDial v1.0 validation dense ground "
        "truth annotations.",
    )
    parser.add_argument(
        "--train-dense-json",
        default="data/visdial_1.0_train_dense_annotations.json",
        help="Path to json file containing VisDial v1.0 validation dense ground "
        "truth annotations.",
    )
    parser.add_argument(
        "--augment-train-dense-json",
        default="data/visdial_1.0_train_dense_annotations.json",
        help="Path to json file containing VisDial v1.0 validation dense ground "
        "truth annotations.",
    )


    parser.add_argument(
        "--data_dir",
        default="data/",
        help="Path to data directory.",
    )

    parser.add_argument_group(
        "Arguments independent of experiment reproducibility"
    )
    parser.add_argument(
        "--gpu-ids",
        nargs="+",
        type=int,
        default=0,
        help="List of ids of GPUs to use.",
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=4,
        help="Number of CPU workers for dataloader.",
    )
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Overfit model on 5 examples, meant for debugging.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Whether to validate on val split after every epoch.",
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Load the whole dataset and pre-extracted image features in memory. "
        "Use only in presence of large RAM, atleast few tens of GBs.",
    )

    parser.add_argument_group("Checkpointing related arguments")
    parser.add_argument(
        "--save-dirpath",
        default="checkpoints/",
        help="Path of directory to create checkpoint directory and save "
        "checkpoints.",
    )
    parser.add_argument(
        "--load-pthpath",
        default="",
        help="To continue training, path to .pth file of saved checkpoint.",
    )

    parser.add_argument(
        "--use_pretrained_emb",
        action="store_true",
        help="If we want to use pre-trained embeddings such as BERT.",
    )

    # parser.add_argument(
    #     "--train_qa_emb_file_path",
    #     default="visdial_1.0_train_qac_bert_emb.h5",  #""/visdial_1.0_train_emb.h5",
    #     help="Path to qac embeddings.",
    # )
    #
    # parser.add_argument(
    #     "--val_qa_emb_file_path",
    #     default="visdial_1.0_val_qac_bert_emb.h5",  #""/visdial_1.0_train_emb.h5",
    #     help="Path to qac embeddings.",
    # )
    parser.add_argument(
        "--hist_emb_file_path",
        default="visdial_1.0_test_emb.h5",
        help="Path to hist embeddings.",
    )
    parser.add_argument(
        "--emb_type",
        default="bert",
        help="Type of embeddings to use",
    )

    # SA: arguments related to finetuning using dense annotations
    parser.add_argument(
        "--load_finetune_pthpath",
        default="",
        help="To continue training, path to .pth file of saved checkpoint.",
    )
    parser.add_argument('--min_lr', default=5e-6, type=float,
                        help='Minimum learning rate used by '
                             'ReduceLR scheduler')
    # SA: phase for training/finetuning
    parser.add_argument(
        "--phase",
        default="training",
        choices=["training", "finetuning", "both", "dense", "dense_scratch_train"],
        help="Finetuning using curriculum learning (finetuning)"
    )

    parser.add_argument(
        "--dense_annotation_type",
        default="default",
        choices=["gt_1", "uniform"],
        help="Finetuning using curriculum learning (finetuning). Annotation type."
    )

    # Default doesnt ignore caption. action: store_false
    # To be compatible with previous code
    parser.add_argument(
        "--ignore_caption",
        action="store_false",
        help="If caption should be used as part of history"
    )

    # SA: lr scheduler type
    parser.add_argument(
        "--lr_scheduler_type",
        default="lambda_lr",
        choices=["lambda_lr", "reduce_lr_on_plateau"],
        help="LR scheduler function. Default LambdaLR pytorch method"
    )
    parser.add_argument(
        "--finetune_lr_scheduler_type",
        default="reduce_lr_on_plateau",
        choices=["lambda_lr", "reduce_lr_on_plateau"],
        help="LR scheduler function. Default Reduce LR on plateau pytorch method"
    )
    parser.add_argument(
        "--dense_regression",
        action="store_true",
        help="If we want to use regression instead of classification for dense annotations",
    )

    args = parser.parse_args()
    return args


# =============================================================================
#   SETUP DATASET, DATALOADER, MODEL, CRITERION, OPTIMIZER, SCHEDULER
# =============================================================================

def get_dataloader(config, args,
                   finetune: bool = False,
                   use_augment_dense: bool = False):

    # SA: pin memory for speed up
    # https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723/2
    pin_memory = config["solver"].get("pin_memory", True)
    print(f"Pin memory is set to {pin_memory}")

    # This should be emb dir.
    emb_dir_file_path = get_emb_dir_file_path(args.data_dir, args.emb_type)
    # SA: todo should be emb_dir_file_path
    # config["dataset"]["qa_emb_file_path"] = "{}/{}".format(args.data_dir, args.qa_emb_file_path)
    # config["dataset"]["hist_emb_file_path"] = "{}/{}".format(args.data_dir, args.hist_emb_file_path)

    # =============================================================================
    #   SETUP DATASET, DATALOADER, MODEL, CRITERION, OPTIMIZER, SCHEDULER
    # =============================================================================

    qa_emb_train_file_path = get_qa_embeddings_file_path(args.data_dir,
                                                         data_type="train",
                                                         emb_type=args.emb_type)
    qa_emb_val_file_path = get_qa_embeddings_file_path(args.data_dir,
                                                       data_type="val",
                                                       emb_type=args.emb_type)
    print(f"Embedding file path for train: {qa_emb_train_file_path}")
    print(f"Embedding file path for valid: {qa_emb_val_file_path}")

    hist_emb_train_file_path = get_hist_embeddings_file_path(emb_dir_file_path,
                                                             data_type="train",
                                                             concat=config["dataset"]["concat_history"],
                                                             emb_type=args.emb_type)
    hist_emb_val_file_path = get_hist_embeddings_file_path(emb_dir_file_path,
                                                             data_type="val",
                                                             concat=config["dataset"]["concat_history"],
                                                             emb_type=args.emb_type)


    pin_memory = config["solver"].get("pin_memory", True)
    print(f"Pin memory is set to {pin_memory}")

    if use_augment_dense:
        augment_dense_annotations_jsonpath=args.augment_train_dense_json
    else:
        augment_dense_annotations_jsonpath=None

    # SA: todo generalize "disc"   config["model"]["decoder"] == "disc"
    train_dataset = VisDialDataset(
        config["dataset"],
        args.train_json,
        args.train_dense_json,
        augment_dense_annotations_jsonpath=augment_dense_annotations_jsonpath,
        qa_emb_file_path=qa_emb_train_file_path,
        hist_emb_file_path=hist_emb_train_file_path,
        use_pretrained_emb=args.use_pretrained_emb,
        use_caption=args.ignore_caption,
        finetune=finetune,
        overfit=args.overfit,
        in_memory=args.in_memory,
        num_workers=args.cpu_workers,
        return_options=True if config["model"]["decoder"] != "gen" else False,
        add_boundary_toks=False if config["model"]["decoder"] != "gen" else True
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["solver"]["batch_size"],
        num_workers=args.cpu_workers,
        shuffle=True,
        pin_memory=pin_memory
    )

    val_dataset = VisDialDataset(
        config["dataset"],
        args.val_json,
        args.val_dense_json,
        qa_emb_file_path=qa_emb_val_file_path,
        hist_emb_file_path=hist_emb_val_file_path,
        use_pretrained_emb=args.use_pretrained_emb,
        use_caption=args.ignore_caption,
        finetune=finetune,
        overfit=args.overfit,
        in_memory=args.in_memory,
        num_workers=args.cpu_workers,
        return_options=True,
        add_boundary_toks=False if config["model"]["decoder"] != "gen" else True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["solver"]["batch_size"]
        if config["model"]["decoder"] != "gen"
        else 5,
        num_workers=args.cpu_workers,
        pin_memory=pin_memory
    )

    # SA: best practice to return dic instead of variables
    dataloader_dic = {
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset
    }

    return dataloader_dic



def get_model(config, args, train_dataset, device):

    # Pass vocabulary to construct Embedding layer.
    encoder = Encoder(config["model"], train_dataset.vocabulary)
    decoder = Decoder(config["model"], train_dataset.vocabulary)
    print("Encoder: {}".format(config["model"]["encoder"]))
    print("Decoder: {}".format(config["model"]["decoder"]))

    # New: Initializing word_embed using GloVe
    if "glove_npy" in config["dataset"]:
        encoder.word_embed.weight.data = torch.from_numpy(np.load(config["dataset"]["glove_npy"]))
        print("Loaded glove vectors from {}".format(config["dataset"]["glove_npy"]))

    # Share word embedding between encoder and decoder.
    if encoder.word_embed and decoder.word_embed:
        decoder.word_embed = encoder.word_embed

    # Wrap encoder and decoder in a model.
    model = EncoderDecoderModel(encoder, decoder).to(device)
    if -1 not in args.gpu_ids:
        model = nn.DataParallel(model, args.gpu_ids)
    return model


def compute_ndcg_type_loss(output, labels, log_softmax=nn.LogSoftmax(dim=-1)):
    """
    Refer https://discuss.pytorch.org/t/multi-class-cross-entropy-loss-and-softmax-in-pytorch/24920
    https://discuss.pytorch.org/t/multi-class-cross-entropy-loss-function-implementation-in-pytorch/19077/47
    https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/loss.py

    :param labels:  (bs, options)
    :param preds:  (bs, options)
    :return:
    """

    # torch._C._nn.nll_loss
    # RuntimeError: Expected object of scalar type Long
    # but got scalar type Float for argument #2 'target'

    # Or try https://pytorch.org/docs/master/nn.html#torch.nn.MultiLabelMarginLoss
    # https://discuss.pytorch.org/t/softmax-regression-from-scratch-nan-loss/49861/2
    output = log_softmax(output)  # (bs, options)
    # -- we are taking one particular round defined by round_id
    # take log softmax first to normalize the logits
    batch_size, num_options = output.size()
    loss = -torch.mean(torch.sum(labels.view(batch_size, -1) * output.view(batch_size, -1), dim=1))
    return loss

def mse_loss(output, labels, criterion, log_softmax=nn.LogSoftmax(dim=-1)):
    # output = log_softmax(output)  # (bs, options)
    batch_size, num_options = output.size()
    labels = labels.view(batch_size, -1)
    output = output.view(batch_size, -1)
    loss = torch.sum((output - labels) ** 2)
    # loss = criterion(labels, output)
    return loss



def get_loss_criterion(config, train_dataset):
    # Loss function.
    if config["model"]["decoder"] == "disc":
        criterion = nn.CrossEntropyLoss()
    elif config["model"]["decoder"] == "gen":
        criterion = nn.CrossEntropyLoss(
            ignore_index=train_dataset.vocabulary.PAD_INDEX
        )
    else:
        raise NotImplementedError
    return criterion


def get_solver(config, args, train_dataset, val_dataset, model, finetune: bool = False):

    if not finetune:
        initial_lr = config["solver"]["initial_lr"]
        lr_scheduler_type = args.lr_scheduler_type
    else:
        lr_scheduler_type = args.finetune_lr_scheduler_type
        initial_lr = config["solver"]["initial_lr_curriculum"]

    if config["solver"]["training_splits"] == "trainval":
        iterations = (len(train_dataset) + len(val_dataset)) // config["solver"][
            "batch_size"
        ] + 1
    else:
        iterations = len(train_dataset) // config["solver"]["batch_size"] + 1

    def lr_lambda_fun(current_iteration: int) -> float:
        """Returns a learning rate multiplier.

        Till `warmup_epochs`, learning rate linearly increases to `initial_lr`,
        and then gets multiplied by `lr_gamma` every time a milestone is crossed.
        """
        current_epoch = float(current_iteration) / iterations
        if current_epoch <= config["solver"]["warmup_epochs"]:
            alpha = current_epoch / float(config["solver"]["warmup_epochs"])
            return config["solver"]["warmup_factor"] * (1.0 - alpha) + alpha
        else:
            idx = bisect(config["solver"]["lr_milestones"], current_epoch)
            return pow(config["solver"]["lr_gamma"], idx)

    print(f"Initial LR set to: {initial_lr}")
    optimizer = optim.Adamax(model.parameters(), lr=initial_lr)

    # SA: Default setting is lambda_lr
    if lr_scheduler_type == "lambda_lr":
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_fun)
    elif lr_scheduler_type == "reduce_lr_on_plateau":
        # todo SA: be careful: Look for 1 epochs == iterations.
        # We are doing step after each iteration

        # Check tests/reduce_lr_on_plateau.py
        # patience=iterations for train and number for finetune.
        print(f"Total iterations in finetuning would be: {iterations}")
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                   factor=config["solver"]["lr_gamma"],
                                                   patience=0, verbose=True,
                                                   min_lr=args.min_lr)
        assert args.validate, "Current reduce lr on plateau checks the val loss to reduce lr"
        print("NDCG would be used to reduce lr on plateau")
    else:
        raise NotImplementedError

    return optimizer, scheduler, iterations, lr_scheduler_type


def get_batch_criterion_loss_value(config, batch, criterion, output):
    target = (batch["ans_ind"] if config["model"]["decoder"] != "gen"
              else batch["ans_out"])
    batch_loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
    return batch_loss


def train(config, args, dataloader_dic,
          device, finetune: bool = False, load_pthpath: str = "",
          finetune_regression: bool = False,
          dense_scratch_train: bool = False,
          dense_annotation_type: str = "default"):
    """

    :param config:
    :param args:
    :param dataloader_dic:
    :param device:
    :param finetune:
    :param load_pthpath:
    :param finetune_regression:
    :param dense_scratch_train: when we want to start training only on 2000 annotations
    :param dense_annotation_type: default
    :return:
    """
    # =============================================================================
    #   SETUP BEFORE TRAINING LOOP
    # =============================================================================
    train_dataset = dataloader_dic["train_dataset"]
    train_dataloader = dataloader_dic["train_dataloader"]
    val_dataloader = dataloader_dic["val_dataloader"]
    val_dataset = dataloader_dic["val_dataset"]

    model = get_model(config, args, train_dataset, device)

    if finetune and not dense_scratch_train:
        assert load_pthpath != "", "Please provide a path" \
                                        " for pre-trained model before " \
                                        "starting fine tuning"
        print(f"\n Begin Finetuning:")

    optimizer, scheduler, iterations, lr_scheduler_type = get_solver(config, args, train_dataset, val_dataset, model, finetune=finetune)

    start_time = datetime.datetime.strftime(datetime.datetime.utcnow(), '%d-%b-%Y-%H:%M:%S')
    if args.save_dirpath == 'checkpoints/':
        args.save_dirpath += '%s+%s/%s' % (config["model"]["encoder"], config["model"]["decoder"], start_time)
    summary_writer = SummaryWriter(log_dir=args.save_dirpath)
    checkpoint_manager = CheckpointManager(
        model, optimizer, args.save_dirpath, config=config
    )
    sparse_metrics = SparseGTMetrics()
    ndcg = NDCG()
    best_val_loss = np.inf  # SA: initially loss can be any number
    best_val_ndcg = 0.0
    # If loading from checkpoint, adjust start epoch and load parameters.

    # SA: 1. if finetuning -> load from saved model
    # 2. train -> default load_pthpath = ""
    # 3. else load pthpath
    if (not finetune and load_pthpath == "") or dense_scratch_train:
        start_epoch = 1
    else:
        # "path/to/checkpoint_xx.pth" -> xx
        ### To cater model finetuning from models with "best_ndcg" checkpoint
        try:
            start_epoch = int(load_pthpath.split("_")[-1][:-4]) + 1
        except:
            start_epoch = 1

        model_state_dict, optimizer_state_dict = load_checkpoint(load_pthpath)

        # SA: updating last epoch
        checkpoint_manager.update_last_epoch(start_epoch)

        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(model_state_dict)
        else:
            model.load_state_dict(model_state_dict)

        # SA: for finetuning optimizer should start from its learning rate
        if not finetune:
            optimizer.load_state_dict(optimizer_state_dict)
        else:
            print("Optimizer not loaded. Different optimizer for finetuning.")
        print("Loaded model from {}".format(load_pthpath))

    # =============================================================================
    #   TRAINING LOOP
    # =============================================================================

    # Forever increasing counter to keep track of iterations (for tensorboard log).
    global_iteration_step = (start_epoch - 1) * iterations

    running_loss = 0.0  # New
    train_begin = datetime.datetime.utcnow()  # New

    if finetune:
        end_epoch = start_epoch + config["solver"]["num_epochs_curriculum"]-1
        if finetune_regression:
            # criterion = nn.MSELoss(reduction='mean')
            # criterion = nn.KLDivLoss(reduction='mean')
            criterion = nn.MultiLabelSoftMarginLoss()
    else:
        end_epoch = config["solver"]["num_epochs"]
        # SA: normal training
        criterion = get_loss_criterion(config, train_dataset)

    # SA: end_epoch + 1 => for loop also doing last epoch
    for epoch in range(start_epoch, end_epoch + 1):
        # -------------------------------------------------------------------------
        #   ON EPOCH START  (combine dataloaders if training on train + val)
        # -------------------------------------------------------------------------
        if config["solver"]["training_splits"] == "trainval":
            combined_dataloader = itertools.chain(train_dataloader, val_dataloader)
        else:
            combined_dataloader = itertools.chain(train_dataloader)

        print(f"\nTraining for epoch {epoch}:")
        for i, batch in enumerate(tqdm(combined_dataloader)):
            for key in batch:
                batch[key] = batch[key].to(device)

            optimizer.zero_grad()
            output = model(batch)

            if finetune:
                target = batch["gt_relevance"]
                # Same as for ndcg validation, only one round is present
                output = output[
                        torch.arange(output.size(0)), batch["round_id"] - 1, :
                    ]
                # SA: todo regression loss
                if finetune_regression:
                    batch_loss = mse_loss(output, target, criterion)
                else:
                    batch_loss = compute_ndcg_type_loss(output, target)
            else:
                batch_loss = get_batch_criterion_loss_value(config, batch, criterion, output)

            batch_loss.backward()
            optimizer.step()

            # --------------------------------------------------------------------
            # update running loss and decay learning rates
            # --------------------------------------------------------------------
            if running_loss > 0.0:
                running_loss = 0.95 * running_loss + 0.05 * batch_loss.item()
            else:
                running_loss = batch_loss.item()

            # SA: lambda_lr was configured to reduce lr after milestone epochs
            if lr_scheduler_type == "lambda_lr":
                scheduler.step(global_iteration_step)

            global_iteration_step += 1

            if global_iteration_step % 100 == 0:
                # print current time, running average, learning rate, iteration, epoch
                print("[{}][Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][lr: {:8f}]".format(
                    datetime.datetime.utcnow() - train_begin, epoch,
                        global_iteration_step, running_loss,
                        optimizer.param_groups[0]['lr']))

                # tensorboardX
                summary_writer.add_scalar(
                    "train/loss", batch_loss, global_iteration_step
                )
                summary_writer.add_scalar(
                    "train/lr", optimizer.param_groups[0]["lr"], global_iteration_step
                )
        torch.cuda.empty_cache()


        # -------------------------------------------------------------------------
        #   ON EPOCH END  (checkpointing and validation)
        # -------------------------------------------------------------------------
        if not finetune:
            checkpoint_manager.step(epoch=epoch)
        else:
            print("Validating before checkpointing.")

        # SA: ideally another function: too much work
        # Validate and report automatic metrics.
        if args.validate:

            # Switch dropout, batchnorm etc to the correct mode.
            model.eval()
            val_loss = 0

            print(f"\nValidation after epoch {epoch}:")
            for i, batch in enumerate(tqdm(val_dataloader)):
                for key in batch:
                    batch[key] = batch[key].to(device)
                with torch.no_grad():
                    output = model(batch)
                    if finetune:
                        target = batch["gt_relevance"]
                        # Same as for ndcg validation, only one round is present
                        out_ndcg = output[
                                torch.arange(output.size(0)), batch["round_id"] - 1, :
                            ]
                        # SA: todo regression loss
                        if finetune_regression:
                            batch_loss = mse_loss(out_ndcg, target, criterion)
                        else:
                            batch_loss = compute_ndcg_type_loss(out_ndcg, target)
                    else:
                        batch_loss = get_batch_criterion_loss_value(config, batch, criterion, output)

                    val_loss += batch_loss.item()
                sparse_metrics.observe(output, batch["ans_ind"])
                if "gt_relevance" in batch:
                    output = output[
                        torch.arange(output.size(0)), batch["round_id"] - 1, :
                    ]
                    ndcg.observe(output, batch["gt_relevance"])

            all_metrics = {}
            all_metrics.update(sparse_metrics.retrieve(reset=True))
            all_metrics.update(ndcg.retrieve(reset=True))
            for metric_name, metric_value in all_metrics.items():
                print(f"{metric_name}: {metric_value}")
            summary_writer.add_scalars(
                "metrics", all_metrics, global_iteration_step
            )

            model.train()
            torch.cuda.empty_cache()

            val_loss = val_loss / len(val_dataloader)
            print(f"Validation loss for {epoch} epoch is {val_loss}")
            print(f"Validation loss for batch is {batch_loss}")

            summary_writer.add_scalar(
                "val/loss", batch_loss, global_iteration_step
            )

            if val_loss < best_val_loss:
                print(f" Best model found at {epoch} epoch! Saving now.")
                best_val_loss = val_loss
                if dense_annotation_type == "default":
                    checkpoint_manager.save_best()
            else:
                print(f" Not saving the model at {epoch} epoch!")

            # SA: Saving the best model both for loss and ndcg now
            val_ndcg = all_metrics["ndcg"]
            if val_ndcg > best_val_ndcg:
                print(f" Best ndcg model found at {epoch} epoch! Saving now.")
                best_val_ndcg = val_ndcg
                if dense_annotation_type == "default":
                    checkpoint_manager.save_best(ckpt_name="best_ndcg")
                else:
                    # SA: trying for dense annotations
                    ckpt_name = f"best_ndcg_annotation_{dense_annotation_type}"
                    checkpoint_manager.save_best(ckpt_name=ckpt_name)
            else:
                print(f" Not saving the model at {epoch} epoch!")

            # SA: "reduce_lr_on_plateau" works only with validate for now
            if lr_scheduler_type == "reduce_lr_on_plateau":
                # scheduler.step(val_loss)
                # SA: # Loss should decrease while ndcg should increase!
                # can also change the mode in LR reduce on plateau to max
                scheduler.step(-1*val_ndcg)

def main():
    args = parse_args()
    print("Starting the model run now: ", datetime.datetime.now())
    # For reproducibility.
    # Refer https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    # =============================================================================
    #   INPUT ARGUMENTS AND CONFIG
    # =============================================================================

    # keys: {"dataset", "model", "solver"}
    config = yaml.load(open(args.config_yml))

    # SA: Changing relative paths to absolute paths
    # Works when you also provide data paths as "../data/features****"
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
    # SA: confirm this is working.
    # see: https://gist.github.com/shubhamagarwal92/8ecf839cf70c4990e3540d0bb4f288ff
    torch.cuda.set_device(device)

    print("Running on:", args.gpu_ids)
    print("First gpu id", args.gpu_ids[0])
    print("Verifying device", device)

    # Print config and args.
    print(yaml.dump(config, default_flow_style=False))
    for arg in vars(args):
        print("{:<20}: {}".format(arg, getattr(args, arg)))

    print("Training phase from the python code: ",args.phase)
    # Normal training
    if args.phase in ["training", "both"]:
        print("Starting training")
        dataloader_dic = get_dataloader(config, args, finetune=False)
        train(config, args, dataloader_dic, device, finetune=False, load_pthpath=args.load_pthpath)

    # Sequential since you can ask for both train and finetune
    if args.phase in ["finetuning", "both"]:
        print("Starting finetuning")
        # SA: todo checkpoint path
        dataloader_dic = get_dataloader(config, args, finetune=True)

        if args.dense_annotation_type == "default":
            train(config, args, dataloader_dic, device, finetune=True, load_pthpath=args.load_finetune_pthpath,
                  finetune_regression=args.dense_regression)
        else:
            train(config, args, dataloader_dic, device, finetune=True, load_pthpath=args.load_finetune_pthpath,
                  finetune_regression=args.dense_regression, dense_annotation_type=args.dense_annotation_type)


    # Train only on dense annotations
    if args.phase in ["dense_scratch_train"]:
        print("Starting finetuning")
        # SA: todo checkpoint path
        dataloader_dic = get_dataloader(config, args, finetune=True)
        train(config, args, dataloader_dic, device, finetune=True,
              dense_scratch_train=True)


    print("Training done! Time: ", datetime.datetime.utcnow())


if __name__ == "__main__":
    main()
