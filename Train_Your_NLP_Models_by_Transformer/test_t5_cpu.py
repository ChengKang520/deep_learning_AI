

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('./workspace/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Import libraries
import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def get_dataset(tokenizer, type_path, args):
    return TweetDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path,  max_len=args.max_seq_length)

## Model
class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.hparams = hparams

        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)

    def is_logger(self):
        return self.trainer.global_rank <= 0

    def forward(
            self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        labels = batch["target_ids"]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
    #     if self.trainer.use_tpu:
    #         # xm.optimizer_step(optimizer)
    #         ha = 1
    #     else:
    #         optimizer.step()
    #     optimizer.zero_grad()
    #     self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True,
                                num_workers=4)
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

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="val", args=self.hparams)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)


logger = logging.getLogger(__name__)

class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))






import csv
from dataclasses import dataclass

from enum import Enum
from typing import List, Optional
from transformers import PreTrainedTokenizer


@dataclass(frozen=True)
class InputExample:
    example_id: str
    text: str
    sentiment: str
    label: str
    """
      A single training/test example for multiple choice
      Args:
          example_id: Unique id for the example.
          contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
          answer : str containing answer for which we need to generate question
          label: string containg questions
      """


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class SwagProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "val.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        raise ValueError(
            "For swag testing, the input file does not contain a label column. It can not be tested in current code"
            "setting!"
        )
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def _read_csv(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f))

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        if type == "train" and lines[0][2] != "selected_text":
            raise ValueError("For training, the input file must contain a label column.")

        examples = [
            InputExample(
                example_id=line[0],
                # common beginning of each
                # choice is stored in "sent2".
                text=line[1],
                sentiment=line[3],
                label=line[2]

            )
            for line in lines[1:]  # we skip the line with the column names
        ]

        return examples

class TweetDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=512):
        self.data_dir = data_dir
        self.type_path = type_path
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self.proc = SwagProcessor()

        self._build()

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def __len__(self):
        return len(self.inputs)

    def _build(self):
        if self.type_path == 'train':
            examples = self.proc.get_train_examples(self.data_dir)
        else:
            examples = self.proc.get_dev_examples(self.data_dir)

        for example in examples:
            self._create_features(example)

    def _create_features(self, example):
        input_ = example.text
        answer = example.sentiment
        input_ = "text: %s  sentiment: %s </s>" % (input_, answer)
        target = example.label
        target = "%s </s>" % (str(target))

        # tokenize inputs
        tokenized_inputs = self.tokenizer.batch_encode_plus(
            [input_], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt", truncation=True
        )
        # tokenize targets
        tokenized_targets = self.tokenizer.batch_encode_plus(
            [target], max_length=150, pad_to_max_length=True, return_tensors="pt", truncation=True
        )

        self.inputs.append(tokenized_inputs)
        self.targets.append(tokenized_targets)


tokenizer = T5Tokenizer.from_pretrained('t5-small')

dataset = TweetDataset(tokenizer, data_dir='./workspace/input/tweetextract/', type_path='val')
# print('********************************************')
# len(dataset)
# print('********************************************')

# data = dataset[69]
# print(tokenizer.decode(data['source_ids']))
# print(tokenizer.decode(data['target_ids']))


args_dict = dict(
    data_dir="", # path for data files
    output_dir="", # path to save the checkpoints
    model_name_or_path='t5-small',
    tokenizer_name_or_path='t5-small',
    max_seq_length=512,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=8,
    eval_batch_size=8,
    num_train_epochs=2,
    gradient_accumulation_steps=16,
    n_gpu=1,
    # early_stop_callback=False,
    fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)

args_dict.update({'data_dir': './workspace/input/tweetextract/',
                  'output_dir': './workspace/working/t5_tweet/',
                  'num_train_epochs': 1})
args = argparse.Namespace(**args_dict)
# print('********************************************')
# print(args_dict)
# print('********************************************')
# print(args.output_dir)
# print('********************************************')

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=5
)
# print(checkpoint_callback)
# print('********************************************')

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=None, #args.n_gpu,
    max_epochs=args.num_train_epochs,
    precision = 16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=checkpoint_callback,
    callbacks=[LoggingCallback()],
)

# train_params = dict(
#     accumulate_grad_batches=args.gradient_accumulation_steps,
# )
model = T5FineTuner(args)


# # device = torch.device("cpu")
# # model = T5FineTuner(args).to(device)
# model = T5FineTuner(args).to(device)
# # print(model)
#
# for param in model.parameters():
#     param.requires_grad = False
#
# ######################################################################
# # 1 and 2. delete the “classifier” part that maps “features” to scores of 1000 ImageNet classes.Consider using torch.nn.
# #          BatchNorm1d (after linear layers) or torch.nn.Dropout (after activations) inside your classifier block.
# # num_ftrs = model.model.encoder.block[0].layer[0]
#
#
# from transformers.models.t5.modeling_t5 import T5Block
# from transformers.models.t5.modeling_t5 import T5Stack
# from transformers.models.t5.configuration_t5 import T5Config
#
#
#
# decoder_config = T5Config()
# decoder_config.is_decoder = True
# decoder_config.is_encoder_decoder = False
# decoder_config.num_layers = 1
# decoder_config.relative_attention_num_buckets = False
#
# # embedded_last = torch.nn.Embedding(decoder_config.vocab_size, decoder_config.d_model).to(device)
# # t5_stack_last = T5Stack(decoder_config)
#
# t5_block_last = T5Block(decoder_config)
#
#
# num_ftrs_in = model.model.lm_head.in_features
# num_ftrs_out = model.model.lm_head.out_features
# net_sequence = torch.nn.Sequential(torch.nn.Conv1d(in_channels=num_ftrs_in, out_channels=num_ftrs_out, kernel_size=1),
#                                 torch.nn.BatchNorm1d(num_ftrs_out),
#                                 torch.nn.ReLU()
#                                 )
#
# t5_block_last.add_module('conv_sequence', net_sequence)
# t5_block_last.add_module('conv_sequence', torch.nn.Linear(in_features=512, out_features=32128, bias=False))
#
# model.lm_head = t5_block_last
# print(model.lm_head)
#
#
# # traced_model = torch.jit.trace(model, (input_ids, attention_mask, decoder_input_ids))
# # from torch.utils.tensorboard import SummaryWriter
# # with SummaryWriter(comment='T5_small')as w:
# #     # w.add_embedding(dummy_input, metadata=None, label_img=None, global_step=None, tag='default', metadata_header=None)
# #     # w.add_graph(traced_model, (input_ids, attention_mask, decoder_input_ids))
# #     w.add_graph(traced_model, (input_ids,))
# # # !tensorboard --logdir=runs
#
#
#
# input_ids = torch.randint(0, 1, (1, 512))
# attention_mask = torch.randint(0, 1, (1, 512))
# decoder_input_ids = torch.randint(0, 1, (1, 512))
#

if __name__ == '__main__':
    # import sys
    # if sys.platform == 'win32' and getattr(sys, 'frozen', False):
    #     from multiprocessing.forking import freeze_support
    #     freeze_support()
    # print('********************************************')
    # print(sys.platform)

    trainer = pl.Trainer(**train_params)
    torch.multiprocessing.freeze_support()
    print('********************************************')
    trainer.fit(model)
    ha = 1


