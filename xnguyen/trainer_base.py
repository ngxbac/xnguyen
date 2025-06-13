import torch
import numpy as np
import os
import datetime
import math
import sys
import time
import json
from pathlib import Path
from typing import List
from abc import ABC, abstractmethod
import torch.nn as nn

from xnguyen.scheduler import OneCycleLRWithWarmup
import torch.distributed as dist
from xnguyen.utils import MetricLogger

from accelerate import Accelerator, DeepSpeedPlugin
from timm.utils import ModelEmaV3


class AcceleratorTrainer:
    def __init__(self, args):
        self.args = args
        self.accelerator = self.init_accelerator()
        self.pre_init()
        self.train_loader, self.valid_loader = self.get_dataloader()
        self.model = self.get_model()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()
        self.criterion = self.get_criterion()

        self.accumulation_steps = getattr(self.args, "gradient_accumulation_steps", 1)

        self.prepare_multi_gpu()
        # self.setup_metric_comparision(score_key="loss", compare_fn="decrease")

        Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)

        # EMA support
        self.ema_decay = getattr(self.args, "ema_decay", None)
        self.ema_model = None
        if self.ema_decay is not None:
            print(f"[EMA] Initializing ModelEmaV2 with decay = {self.ema_decay}")
            self.ema_model = ModelEmaV3(
                self.model,
                decay=self.ema_decay,
                device="cpu" if getattr(self.args, "ema_on_cpu", False) else None,
            )

        # Early stopping
        self.early_stopping_patience = getattr(
            self.args, "early_stopping_patience", None
        )
        self.early_stopping_counter = 0

        # Setup metric dicts for regular model and EMA model
        self.setup_metric_comparision(score_key="loss", compare_fn="decrease")

    def __del__(self):
        try:
            torch.cuda.empty_cache()
            if hasattr(self, "accelerator"):
                try:
                    self.accelerator.free_memory()
                except Exception as e:
                    print(f"[WARNING] Error calling accelerator.free_memory(): {e}")
            if hasattr(self, "model"):
                del self.model
            if hasattr(self, "optimizer"):
                del self.optimizer
        except:
            pass

    def setup_metric_comparision(self, score_key, compare_fn="increase"):
        # Always store current score key and compare fn
        self._current_score_key = score_key
        self._current_compare_fn = compare_fn

        # Build regular metric dict
        if not (isinstance(score_key, list) and isinstance(compare_fn, list)):
            score_key = [score_key]
            compare_fn = [compare_fn]

        metric_comp_dict = {}
        for score_key_, compare_fn_ in zip(score_key, compare_fn):
            metric_comp_dict_ = self._setup_metric_comparision(
                score_key=score_key_, compare_fn=compare_fn_
            )
            metric_comp_dict.update(metric_comp_dict_)

        # Update model metric dict
        self.metric_comp_dict = metric_comp_dict
        self.metric_comp_dict_model = metric_comp_dict.copy()

        # If EMA is enabled → update its metric dict too
        if self.ema_model is not None:
            metric_comp_dict_ema = {}
            for score_key_, compare_fn_ in zip(score_key, compare_fn):
                metric_comp_dict_ema_ = self._setup_metric_comparision(
                    score_key=score_key_, compare_fn=compare_fn_
                )
                metric_comp_dict_ema.update(metric_comp_dict_ema_)
            self.metric_comp_dict_ema = metric_comp_dict_ema

    def _setup_metric_comparision(self, score_key, compare_fn="increase"):
        assert compare_fn in ["increase", "decrease"]
        metric_comp_dict = {}
        metric_comp_dict[score_key] = {}
        if compare_fn == "increase":
            best_score = -np.inf
            compare_fn = lambda x1, x2: x1 > x2
        else:
            best_score = np.inf
            compare_fn = lambda x1, x2: x1 < x2

        metric_comp_dict[score_key]["best_score"] = best_score
        metric_comp_dict[score_key]["compare_fn"] = compare_fn
        metric_comp_dict[score_key]["is_best"] = False
        return metric_comp_dict

    def init_accelerator(self):
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
        torch.cuda.set_device(local_rank)

        return accelerator

    def setup_for_distributed(self, is_master):
        import builtins

        builtin_print = builtins.print

        def print(*args, **kwargs):
            force = kwargs.pop("force", False)
            if is_master or force:
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
        valid_stats_model = None
        valid_stats_ema = None

        if self.valid_loader and (epoch + 1) % self.args.eval_interval == 0:
            # Validate regular model
            valid_stats_model = self.run_one_epoch(
                self.valid_loader, epoch, is_train=False
            )

            # Validate EMA model
            if self.ema_model is not None:
                print("[EMA] Validating EMA model")
                orig_model = self.model
                self.model = self.ema_model.module
                valid_stats_ema = self.run_one_epoch(
                    self.valid_loader, epoch, is_train=False
                )
                self.model = orig_model

        return valid_stats_model, valid_stats_ema

    def train_one_epoch(self, epoch):
        if self.args.distributed:
            self.train_loader.set_epoch(epoch)
        train_stats = self.run_one_epoch(self.train_loader, epoch, is_train=True)
        return train_stats

    def _is_best(self, stats, score_key, best_score, compare_fn):
        current_score = stats[score_key]
        is_best = False
        if compare_fn(current_score, best_score):
            best_score = current_score
            is_best = True
        return is_best, best_score

    def check_best_score(self, stats):
        if stats is None:
            return stats
        for score_key in self.metric_comp_dict.keys():
            is_best, new_best_score = self._is_best(
                stats=stats,
                score_key=score_key,
                best_score=self.metric_comp_dict[score_key]["best_score"],
                compare_fn=self.metric_comp_dict[score_key]["compare_fn"],
            )
            self.metric_comp_dict[score_key]["is_best"] = is_best
            if is_best:
                self.metric_comp_dict[score_key]["best_score"] = new_best_score
            stats[f"best_{score_key}"] = self.metric_comp_dict[score_key]["best_score"]
        return stats

    def get_model_state_dict(self):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        state_dict = unwrapped_model.state_dict()
        del unwrapped_model
        return state_dict

    def save_hf_model(self, tag):
        try:
            from diffusers.models.modeling_utils import ModelMixin

            if isinstance(self.model, ModelMixin):
                save_dir = os.path.join(self.args.output_dir, "hf")
                self.model.save_pretrained(f"{save_dir}/{tag}/")
        except:
            return

    def save_ckpt(self, tag, epoch, best_score=None, save_model=True, save_ema=True):
        if self.accelerator.is_main_process:
            # Save regular model
            if save_model:
                self.save_hf_model(tag)
                ckpt_path = os.path.join(self.args.output_dir, f"{tag}.pth")
                state_dict = self.get_model_state_dict()
                try:
                    torch.save(
                        {
                            "epoch": epoch,
                            "best_score": best_score,
                            "args": self.args,
                            "model": state_dict,
                        },
                        ckpt_path,
                    )
                    print(f"[SAVE] Model checkpoint saved to {ckpt_path}")
                except Exception as e:
                    print(
                        f"[SAVE WARNING] Could not save model checkpoint to {ckpt_path}: {e}"
                    )

            # Save EMA model
            if save_ema and self.ema_model is not None:
                ema_ckpt_path = os.path.join(self.args.output_dir, f"{tag}_ema.pth")
                try:
                    ema_state_dict = self.ema_model.module.state_dict()
                    torch.save(ema_state_dict, ema_ckpt_path)
                    print(f"[SAVE] EMA checkpoint saved to {ema_ckpt_path}")
                except Exception as e:
                    print(
                        f"[SAVE WARNING] Could not save EMA checkpoint to {ema_ckpt_path}: {e}"
                    )

            # Save full accelerator state
            state_path = os.path.join(self.args.output_dir, tag)
            try:
                self.accelerator.save_state(state_path)
                print(f"[SAVE] Accelerator state saved to {state_path}")
            except Exception as e:
                print(
                    f"[SAVE WARNING] Could not save Accelerator state to {state_path}: {e}"
                )

        self.accelerator.wait_for_everyone()

    def save(self, epoch):
        # Save last checkpoint for regular model
        self.metric_comp_dict = self.metric_comp_dict_model
        self.save_ckpt("last", epoch, None, save_model=True, save_ema=False)

        # Save best regular model
        for score_key in self.metric_comp_dict.keys():
            if self.metric_comp_dict[score_key]["is_best"]:
                best_score = self.metric_comp_dict[score_key]["best_score"]
                print(
                    f"Saving best {score_key} at epoch ({epoch}) with score: {best_score}"
                )
                self.save_ckpt(
                    f"best_{score_key}",
                    epoch,
                    best_score,
                    save_model=True,
                    save_ema=False,
                )

        # Save best EMA model
        if self.ema_model is not None:
            self.metric_comp_dict = self.metric_comp_dict_ema
            for score_key in self.metric_comp_dict.keys():
                if self.metric_comp_dict[score_key]["is_best"]:
                    best_score = self.metric_comp_dict[score_key]["best_score"]
                    print(
                        f"[EMA] Saving best {score_key} at epoch ({epoch}) with score: {best_score}"
                    )
                    self.save_ckpt(
                        f"best_{score_key}_ema",
                        epoch,
                        best_score,
                        save_model=False,
                        save_ema=True,
                    )

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
        if not os.path.isfile(ckp_path):
            return
        print("Found checkpoint at {}".format(ckp_path))
        checkpoint = torch.load(ckp_path, map_location="cpu")

        for key, value in kwargs.items():
            if key in checkpoint and value is not None:
                try:
                    msg = value.load_state_dict(checkpoint[key], strict=True)
                    print(
                        f"=> loaded '{key}' from checkpoint '{ckp_path}' with msg {msg}"
                    )
                except TypeError:
                    try:
                        msg = value.load_state_dict(checkpoint[key])
                        print(f"=> loaded '{key}' from checkpoint: '{ckp_path}'")
                    except ValueError:
                        print(
                            f"=> failed to load '{key}' from checkpoint: '{ckp_path}'"
                        )
            else:
                print(f"=> key '{key}' not found in checkpoint: '{ckp_path}'")

        if run_variables is not None:
            for var_name in run_variables:
                if var_name in checkpoint:
                    run_variables[var_name] = checkpoint[var_name]

    # === NEW HOOKS ===
    def before_train_epoch(self, epoch):
        """Hook called before starting train_one_epoch"""
        pass

    def after_train_epoch(self, epoch, train_stats, valid_stats):
        """Hook called after finishing one epoch (train+valid), before saving/checkpoint"""
        pass

    def train(self):
        start_time = time.time()
        to_restore = {"epoch": 0}
        if os.path.isfile(self.args.resume):
            self.restart_from_checkpoint(
                self.args.resume,
                model=self.model,
                run_variables=to_restore,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )

        start_epoch = to_restore["epoch"]
        print(f"[+] Start training !")
        for epoch in range(start_epoch, self.args.epochs):
            self.before_train_epoch(epoch)

            train_stats = self.train_one_epoch(epoch)
            valid_stats_model, valid_stats_ema = self.valid_one_epoch(epoch)

            # Choose stats for early stopping (based on regular model)
            stats_for_early_stopping = (
                valid_stats_model if valid_stats_model is not None else train_stats
            )

            # Check best REGULAR model
            self.metric_comp_dict = self.metric_comp_dict_model
            self.check_best_score(stats_for_early_stopping)

            # Check best EMA model
            if self.ema_model is not None and valid_stats_ema is not None:
                self.metric_comp_dict = self.metric_comp_dict_ema
                self.check_best_score(valid_stats_ema)

            # Save checkpoints
            self.save(epoch=epoch)

            # Save logs
            self.save_log(
                train_stats=train_stats, valid_stats=valid_stats_model, epoch=epoch
            )
            if valid_stats_ema is not None:
                # Optionally you can suffix EMA keys here if you want
                self.save_log(
                    train_stats=train_stats, valid_stats=valid_stats_ema, epoch=epoch
                )

            # Early stopping based on regular model
            improved = any(
                [
                    self.metric_comp_dict_model[k]["is_best"]
                    for k in self.metric_comp_dict_model.keys()
                ]
            )

            if improved:
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            if self.early_stopping_patience is not None:
                print(
                    f"[EARLY STOPPING] Counter = {self.early_stopping_counter} / {self.early_stopping_patience}"
                )
                if self.early_stopping_counter >= self.early_stopping_patience:
                    print(
                        f"[EARLY STOPPING] No improvement for {self.early_stopping_counter} epochs. Stopping training."
                    )
                    break

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def model_forward(self, batch, is_train=True):
        with torch.amp.autocast(device_type="cuda", enabled=self.args.use_fp16):
            if not is_train:
                with torch.no_grad():
                    output_dict = self.model(batch)
            else:
                output_dict = self.model(batch)
            return output_dict

    def pre_forward(self, batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = batch[k].cuda()
        return batch

    def pos_forward(self, batch, otput_dict):
        return batch, output_dict

    def backprobagation(self, loss, is_train=True):
        if not math.isfinite(loss.item()):
            self.accelerator.print(
                f"Loss is {loss.item()}, stopping training", force=True
            )
            sys.exit(1)

        if not is_train:
            return

        self.accelerator.backward(loss)

        if (getattr(self, "epoch_step", 0) + 1) % self.accumulation_steps == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            if self.ema_model is not None:
                self.ema_model.update(self.model)

    def run_one_epoch(self, data_loader, epoch, is_train):
        self.epoch = epoch
        self.epoch_step = 0
        self.model.train() if is_train else self.model.eval()
        prefix = "TRAIN" if is_train else "VALID"

        metric_logger = MetricLogger(delimiter="  ")
        header = "Epoch: [{}/{}]".format(epoch, self.args.epochs)

        for batch in metric_logger.log_every(data_loader, 10, header):
            if is_train:
                self.optimizer.zero_grad()

            batch = self.pre_forward(batch)
            output_dict = self.model_forward(batch=batch, is_train=is_train)
            batch, output_dict = self.pos_forward(batch=batch, output_dict=output_dict)
            ret_dict = self.criterion(output_dict, batch)
            self.backprobagation(loss=ret_dict["loss"], is_train=is_train)

            if self.args.distributed:
                torch.cuda.synchronize()

            if "metrics" in ret_dict:
                metric_logger.update(**ret_dict["metrics"])
            metric_logger.update(loss=ret_dict["loss"].item())
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=self.optimizer.param_groups[0]["weight_decay"])

            self.epoch_step += 1

        if self.args.distributed:
            metric_logger.synchronize_between_processes()

        print(f"[{prefix}] Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
