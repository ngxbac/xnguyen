import torch
import numpy as np
import os
import datetime
import math
import sys
import time
import json
from pathlib import Path
from collections import defaultdict, deque
from typing import List, Optional
from abc import ABC, abstractmethod
import torch.nn as nn

from xnguyen.scheduler import OneCycleLRWithWarmup
import torch.distributed as dist
from xnguyen.utils import MetricLogger


from accelerate import Accelerator, DeepSpeedPlugin


class AcceleratorTrainer:
    def __init__(self, args):
        self.args = args
        self.accelerator = self.init_accelerator()
        # self.scale_lr()
        self.pre_init()
        self.train_loader, self.valid_loader = self.get_dataloader()
        self.model = self.get_model()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()
        self.criterion = self.get_criterion()

        self.prepare_multi_gpu()

        self.setup_metric_comparision(score_key="loss", compare_fn="decrease")

        Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)

    def setup_metric_comparision(self, score_key, compare_fn="increase"):
        assert compare_fn in ["increase", "decrease"]
        if compare_fn == "increase":
            self.best_score = -np.inf
            self.compare_fn = lambda x1, x2: x1 > x2
        else:
            self.best_score = np.inf
            self.compare_fn = lambda x1, x2: x1 < x2

        self.score_key = score_key

    def init_accelerator(self):
        from accelerate import Accelerator
        from accelerate.utils import DistributedDataParallelKwargs

        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_clipping=1.0)
        accelerator = Accelerator(
            split_batches=False,
            kwargs_handlers=[kwargs],
            mixed_precision="fp16" if self.args.use_fp16 else "no",
            deepspeed_plugin=deepspeed_plugin if self.args.deepspeed else None,
        )
        accelerator.print("PID of this process =", os.getpid())
        device = accelerator.device
        accelerator.print("device:", device)
        num_devices = torch.cuda.device_count()
        if num_devices == 0:
            num_devices = 1
        accelerator.print(accelerator.state)
        local_rank = accelerator.state.local_process_index
        world_size = accelerator.state.num_processes
        distributed = not accelerator.state.distributed_type == "NO"
        accelerator.print(
            "distributed =",
            distributed,
            "num_devices =",
            num_devices,
            "local rank =",
            local_rank,
            "world size =",
            world_size,
        )

        self.args.distributed = distributed
        self.setup_for_distributed(accelerator.is_main_process)

        return accelerator

    def setup_for_distributed(self, is_master):
        """
        This function disables printing when not in master process
        """
        import builtins

        builtin_print = builtins.print

        def print(*args, **kwargs):
            force = kwargs.pop("force", False)
            if is_master:  # or force:
                now = datetime.datetime.now().time()
                builtin_print("[{}] ".format(now), end="")  # print with time stamp
                builtin_print(*args, **kwargs)

        builtins.print = print

    def prepare_multi_gpu(self):
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.valid_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
            self.valid_loader,
            self.scheduler,
        )

    def scale_lr(self):
        # num_process = int(os.environ["WORLD_SIZE"])
        # self.args.lr *= (self.args.batch_size * num_process) / 256.0
        self.args.lr *= self.accelerator.num_processes

    def get_dataloader(self):
        raise NotImplementedError

    def get_model(self):
        raise NotImplementedError

    def get_criterion(self):
        raise NotImplementedError

    def pre_init(self):
        pass

    def get_optimizer(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        return optimizer

    def get_scheduler(self):
        # prepare lr scheduler
        one_epoch_steps = len(self.train_loader)
        total_steps = self.args.epochs * one_epoch_steps
        print("one_epoch_steps_per_gpu:", one_epoch_steps)
        print("total_steps:", total_steps)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.args.lr,
            total_steps=total_steps,
            final_div_factor=100,
            last_epoch=-1,
            pct_start=2 / self.args.epochs,
        )

        return scheduler

    def valid_one_epoch(self, epoch):
        if self.valid_loader and (epoch + 1) % self.args.eval_interval == 0:
            valid_stats = self.run_one_epoch(self.valid_loader, epoch, is_train=False)
            valid_stats[f"best_{self.score_key}"] = self.best_score
        else:
            valid_stats = None

        return valid_stats

    def train_one_epoch(self, epoch):
        if self.args.distributed:
            self.train_loader.set_epoch(epoch)
        train_stats = self.run_one_epoch(self.train_loader, epoch, is_train=True)
        return train_stats

    def is_best(self, stats):
        if stats is None:
            return False

        current_score = stats[self.score_key]
        is_best = False
        if self.compare_fn(current_score, self.best_score):
            self.best_score = current_score
            is_best = True

        return is_best

    def save_ckpt(self, tag, epoch):
        if self.accelerator.is_main_process:
            ckpt_path = self.args.output_dir + f"/{tag}.pth"
            unwrapped_model = self.accelerator.unwrap_model(self.model)

            try:
                torch.save(
                    {
                        "epoch": epoch,
                        "args": self.args,
                        "model": unwrapped_model.state_dict(),
                    },
                    ckpt_path,
                )
            except:
                print("Couldn't save... moving on to prevent crashing.")
            del unwrapped_model

        state_path = os.path.join(self.args.output_dir, tag)
        self.accelerator.save_state(state_path)

    def save(self, epoch, is_best):
        self.save_ckpt(f"last", epoch)
        if is_best:
            self.save_ckpt(f"best", epoch)

    def save_log(self, train_stats, valid_stats, epoch):
        def write_log(log_stats):
            if self.accelerator.is_main_process:
                with (Path(self.args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        log_train_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }

        write_log(log_train_stats)

        if valid_stats is not None:
            log_valid_stats = {
                **{f"valid_{k}": v for k, v in valid_stats.items()},
                "epoch": epoch,
            }
            write_log(log_valid_stats)

    def restart_from_checkpoint(self, ckp_path, run_variables=None, **kwargs):
        """
        Re-start from checkpoint
        """
        if not os.path.isfile(ckp_path):
            return
        print("Found checkpoint at {}".format(ckp_path))

        # open checkpoint file
        checkpoint = torch.load(ckp_path, map_location="cpu")

        # key is what to look for in the checkpoint file
        # value is the object to load
        # example: {'state_dict': model}
        for key, value in kwargs.items():
            if key in checkpoint and value is not None:
                try:
                    msg = value.load_state_dict(checkpoint[key], strict=True)
                    print(
                        "=> loaded '{}' from checkpoint '{}' with msg {}".format(
                            key, ckp_path, msg
                        )
                    )
                except TypeError:
                    try:
                        msg = value.load_state_dict(checkpoint[key])
                        print(
                            "=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path)
                        )
                    except ValueError:
                        print(
                            "=> failed to load '{}' from checkpoint: '{}'".format(
                                key, ckp_path
                            )
                        )
            else:
                print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

        # re load variable important for the run
        if run_variables is not None:
            for var_name in run_variables:
                if var_name in checkpoint:
                    run_variables[var_name] = checkpoint[var_name]

    def train(self):
        start_time = time.time()
        to_restore = {"epoch": 0, "best_score": self.best_score}
        if os.path.isfile(self.args.resume):
            self.restart_from_checkpoint(
                self.args.resume,
                model=self.model,
                run_variables=to_restore,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )

        start_epoch = to_restore["epoch"]
        self.best_score = to_restore["best_score"]

        print(f"[+] Start training !")
        for epoch in range(start_epoch, self.args.epochs):
            train_stats = self.train_one_epoch(epoch)
            valid_stats = self.valid_one_epoch(epoch)

            if valid_stats is None:
                valid_stats = train_stats
                stats = train_stats
            else:
                stats = valid_stats

            if self.valid_loader is None:  # No valid steps
                stats = train_stats
            else:
                stats = valid_stats

            is_best = self.is_best(stats)
            self.save(epoch=epoch, is_best=is_best)
            self.save_log(train_stats=train_stats, valid_stats=valid_stats, epoch=epoch)
            self.accelerator.wait_for_everyone()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def model_forward(self, batch, is_train=True):
        if not is_train:
            with torch.no_grad():
                loss = self.model(batch)
        else:
            loss = self.model(batch)

        return loss

    def pre_forward(self, batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = batch[k].cuda()

        return batch

    def pos_forward(self, loss, is_train=True):
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        if not is_train:
            return

        self.accelerator.backward(loss)
        self.optimizer.step()
        self.scheduler.step()
        # self.optimizer.zero_grad()

    def run_one_epoch(self, data_loader, epoch, is_train):
        self.epoch = epoch
        if is_train:
            self.model.train()
            prefix = "TRAIN"
        else:
            self.model.eval()
            prefix = "VALID"

        metric_logger = MetricLogger(delimiter="  ")
        header = "Epoch: [{}/{}]".format(epoch, self.args.epochs)

        for batch in metric_logger.log_every(data_loader, 10, header):
            if is_train:
                self.optimizer.zero_grad()
            batch = self.pre_forward(batch)
            output_dict = self.model_forward(batch=batch, is_train=is_train)
            ret_dict = self.criterion(output_dict, batch)
            self.pos_forward(loss=ret_dict["loss"], is_train=is_train)

            # logging
            if self.args.distributed:
                torch.cuda.synchronize()

            if "metrics" in ret_dict:
                metric_logger.update(**ret_dict["metrics"])
            metric_logger.update(loss=ret_dict["loss"].item())
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=self.optimizer.param_groups[0]["weight_decay"])

        # gather the stats from all processes
        if self.args.distributed:
            metric_logger.synchronize_between_processes()
        print(f"[{prefix}] Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
