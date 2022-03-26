import argparse
import logging
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random
import sys
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelWithLMHead,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from data import WebQATextDataset

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

MODEL_MODES = {
    "base": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
    "pretraining": AutoModelForPreTraining,
    "token-classification": AutoModelForTokenClassification,
    "language-modeling": AutoModelWithLMHead,
    "seq2seq-lm": AutoModelForSeq2SeqLM
}


def set_seed(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class BaseTransformer(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace, mode='base', **config_kwargs):
        """Initialize model"""

        super().__init__()
        self.save_hyperparameters(hparams)
        cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None

        self.config = AutoConfig.from_pretrained(
            self.hparams.config_name if self.hparams.config_name else self.hparams.model_name_or_path,
            cache_dir=cache_dir,
            **config_kwargs,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
            cache_dir=cache_dir,
        )

        self.model = MODEL_MODES[mode].from_pretrained(
            self.hparams.model_name_or_path,
            from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
            config=self.config,
            cache_dir=cache_dir,
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

    def is_logger(self):
        return True

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        model = self.model
        # no_decay = ["bias", "LayerNorm.weight"]
        # optimizer_grouped_parameters = [
        #     {
        #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        #         "weight_decay": self.hparams.weight_decay,
        #     },
        #     {
        #         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        #         "weight_decay": 0.0,
        #     },
        # ]
        # optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        optimizer = AdamW([p for n, p in model.named_parameters()],
                          lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None,
                       using_native_amp=None):
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def get_dataloader(self, filepath: str, batch_size: int, split: str,
                       shuffle: bool = False) -> DataLoader:
        dataset = WebQATextDataset(self.tokenizer, filepath=filepath, split=split,
                                   **self.dataset_kwargs)
        logger.info('loading %s dataloader...', split)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn, shuffle=shuffle,
                                num_workers=4)
        logger.info('done')
        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader(filepath='WebQA_train_val.json', batch_size=self.hparams.train_batch_size,
                                         split="train", shuffle=True)
        t_total = (
                (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(filepath='WebQA_train_val.json', batch_size=self.hparams.eval_batch_size,
                                   split="val")

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader(filepath='WebQA_test.json', batch_size=self.hparams.test_batch_size,
                                   split="test")

    def get_progress_bar_dict(self):
        running_train_loss = self.trainer.running_loss.mean()
        avg_training_loss = running_train_loss.cpu().item() if running_train_loss is not None else float('NaN')
        tqdm_dict = {"Training loss": "{:.3f}".format(avg_training_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def _feature_file(self, mode):
        return os.path.join(
            self.hparams.data_dir,
            "cached_{}_{}_{}".format(
                mode,
                list(filter(None, self.hparams.model_name_or_path.split("/"))).pop(),
                str(self.hparams.max_seq_length),
            ),
        )


class LoggingCallback(pl.Callback):
    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        logger.info("***** Epoch Results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            epoch = metrics['epoch']
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, f"info_{epoch}.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        val = metrics[key]
                        if isinstance(val, torch.Tensor):
                            val = val.cpu().detach().numpy()
                        else:
                            val = str(val)
                        writer.write("{} = {}".format(key, val))
                        writer.write('\n')
                        logger.info("{} = {}".format(key, str(metrics[key])))
            writer.close()

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}".format(key, str(metrics[key])))

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        logger.info("***** Test results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        val = metrics[key]
                        if isinstance(val, torch.Tensor):
                            val = val.cpu().detach().numpy()
                        else:
                            val = str(val)
                        logger.info("{} = {}".format(key, str(metrics[key])))
                        writer.write("{} = {}".format(key, val))
                        writer.write("\n")


def add_args(parser):
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].")
    parser.add_argument("--n_gpu", type=int, default=0)
    parser.add_argument("--n_tpu_cores", type=int, default=0)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--test_batch_size", default=512, type=int)
    parser.add_argument("--max_source_length", default=384, type=int,
                        help="The maximum total input sequence length after tokenization."
                             "Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=512, type=int,
                        help="The maximum total output sequence length.")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the dataset files for the CNN/DM summarization task.")
    parser.add_argument("--early_stopping_patience", type=int, default=-1, required=False,
                        help="-1 means never early stop. early_stopping_patience is measured in validation checks, "
                             "not epochs."
                             "So val_check_interval will effect it.")
    parser.add_argument("--checkpoint", default=None, type=str, help="The checkpoint to initialize model.")
    parser.add_argument("--checkpoint_model", default=None, type=str, help="The checkpoint to initialize model.")
    return parser


def generic_train(model: BaseTransformer, args: argparse.Namespace,
                  early_stopping_callback=False, checkpoint_callback=None):
    # init model
    set_seed(args)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    if not checkpoint_callback:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=1
        )

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        early_stop_callback=early_stopping_callback,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=checkpoint_callback,
        callbacks=[LoggingCallback()],
        log_save_interval=1,
        num_sanity_val_steps=4,
        reload_dataloaders_every_epoch=True
    )

    if args.fp16:
        train_params["use_amp"] = args.fp16
        train_params["amp_level"] = args.fp16_opt_level

    if args.n_tpu_cores > 0:
        global xm

        train_params["num_tpu_cores"] = args.n_tpu_cores
        train_params["gpus"] = 0

    if args.n_gpu > 1:
        train_params["distributed_backend"] = "ddp"

    trainer = pl.Trainer(**train_params)

    if args.do_train:
        trainer.fit(model)

    return trainer

