import argparse
import glob
import logging
import os
import sys
import time

import torch

from data import WebQATextDataset, convert_text
from lightning_base import BaseTransformer, add_args, generic_train

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class BARTQA(BaseTransformer):
    mode = "seq2seq-lm"

    def __init__(self, hparams):
        super().__init__(hparams, mode=self.mode)

        self.step_count = 0

        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            max_target_length=self.hparams.max_target_length
        )
        self.count_valid_epoch = 0

        logger.info("parameters %s", hparams)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, labels=None):
        return self.model(
            input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels)

    def _step(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = batch["source_ids"], batch["source_mask"], batch["target_ids"]
        y_ids = y[:, :-1].contiguous()
        labels = y[:, 1:].clone()
        labels[y[:, 1:] == pad_token_id] = -100
        outputs = self(source_ids, attention_mask=source_mask, decoder_input_ids=y_ids, labels=labels)
        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        # repetition_penalty = 2.5,
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = WebQATextDataset.trim_seq2seq_batch(batch, pad_token_id)
        generated_ids = self.model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            num_beams=5,
            max_length=512,
            length_penalty=5.0,
            early_stopping=True,
            use_cache=True,
        )
        preds = [
            self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in generated_ids
        ]
        target = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
        loss = self._step(batch)
        return {"val_loss": loss, "preds": preds, "target": target}

    def test_step(self, batch, batch_idx):

        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = WebQATextDataset.trim_seq2seq_batch(batch, pad_token_id)
        generated_ids = self.model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            num_beams=5,
            max_length=512,
            length_penalty=5.0,
            early_stopping=True,
            use_cache=True,
        )
        preds = [
            self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in generated_ids
        ]
        target = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
        loss = self._step(batch)

        return {"val_loss": loss, "preds": preds, "target": target}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"val_loss": avg_loss}

    def test_epoch_end(self, outputs):
        if "preds" in outputs[0]:
            output_test_predictions_file = os.path.join(self.hparams.output_dir, "test_predictions_" +
                                                        str(self.count_valid_epoch) + ".txt")
            output_test_targets_file = os.path.join(self.hparams.output_dir, "test_targets_" +
                                                    str(self.count_valid_epoch) + ".txt")
            # write predictions and targets for later rouge evaluation.
            with open(output_test_predictions_file, "w") as p_writer, open(output_test_targets_file,
                                                                           "w") as t_writer:
                for output_batch in outputs:
                    p_writer.writelines(convert_text(s) + "\n" for s in output_batch["preds"])
                    t_writer.writelines(convert_text(s) + "\n" for s in output_batch["target"])
                p_writer.close()
                t_writer.close()
            logger.info("valid epoch: %s", self.count_valid_epoch)
            self.count_valid_epoch += 1
        else:
            logger.info('not in')

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def validation_epoch_end(self, outputs):
        self.step_count += 1

        if "preds" in outputs[0]:
            # output_test_predictions_file = os.path.join(self.hparams.output_dir, "validation_predictions_" +
            #                                             str(self.count_valid_epoch) + ".txt")
            # output_test_targets_file = os.path.join(self.hparams.output_dir, "validation_targets_" +
            #                                         str(self.count_valid_epoch) + ".txt")
            # with open(output_test_predictions_file, "w") as p_writer, open(output_test_targets_file,
            #                                                                "w") as t_writer:
            #     for output_batch in outputs:
            #         p_writer.writelines(convert_text(s) + "\n" for s in output_batch["preds"])
            #         t_writer.writelines(convert_text(s) + "\n" for s in output_batch["target"])
            #     p_writer.close()
            #     t_writer.close()

            logger.info("valid epoch: %s", self.count_valid_epoch)
            avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
            self.count_valid_epoch += 1
        else:
            logger.info('not in')
            avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        metrics = {"val_loss": avg_loss}

        return {"val_loss": avg_loss, "log": metrics}

    def train_epoch_end(self, outputs):
        avg_loss = torch.stack([x["train_loss"] for x in outputs]).mean()
        output_train_info_file = os.path.join(self.hparams.output_dir, "train_info_" +
                                              str(self.count_valid_epoch) + ".txt")
        with open(output_train_info_file, "w") as writer:
            writer.write(avg_loss + "\n")
            writer.close()


def main(args):
    # If output_dir not provided, a folder will be generated in pwd
    if not args.output_dir:
        args.output_dir = os.path.join("./results", f"{time.strftime('%Y%m%d_%H%M%S')}", )
        os.makedirs(args.output_dir)
    model = BARTQA(args)
    if args.checkpoint_model:
        model = model.load_from_checkpoint(args.checkpoint_model)
        logger.info("args.data_dir: %s", args.data_dir)
        model.dataset_kwargs: dict = dict(
            data_dir=args.data_dir,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
        )
        model.hparams = args

    trainer = generic_train(model, args)

    if args.do_predict:
        if args.checkpoint_model:
            trainer.test(model)
        else:
            checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "*.ckpt"), recursive=True)))
            if checkpoints:
                print('Loading weights from {}'.format(checkpoints[-1]))
                model = model.load_from_checkpoint(checkpoints[-1])
                model.dataset_kwargs: dict = dict(
                    data_dir=args.data_dir,
                    max_source_length=args.max_source_length,
                    max_target_length=args.max_target_length,
                )
                model.hparams = args
            trainer.test(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    main(args)

