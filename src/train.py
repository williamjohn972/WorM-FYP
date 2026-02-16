from enum import Enum
from typing import Dict

import torch
import torch.nn as nn
import itertools
import json
import os
from collections import defaultdict

from src.tasks import Tasks
from src.config import Config
from src.model import Show_Task
from src.utils.logger import print_log
from tqdm.auto import tqdm

from src.utils.train_utils import *
from src.utils.metrics_utils import *
from src.utils.viz_utils import viz_results

class Trainer():

    def __init__(self, 
                 model: nn.Module,
                 dataloaders: Dict,
                 config: Config,
                 device:str,
                 logger = None
                 ):
        
        """
        train/val/test_loader: Dict[Task, DataLoader]
        config: Config

        """

        self.model = model

        self.loaders = dataloaders

        self.device = device
        
        self.logger = logger

        # Config
        self.config = config

        # Move the Model to the Device
        model.to(device)

        # Create a Canonical Task Order 
        self.task_list = config.task_config.task_list

        # Create the Optimizer 
        self.optimizer = torch.optim.Adam(params = model.parameters(), 
                                          lr = config.train_config.lr)

        # Create the Scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = self.optimizer, 
                                                                       mode = "min", 
                                                                       factor = 0.8, 
                                                                       patience = 3, 
                                                                    #    verbose = True,
                                                                       threshold = 0.005
                                                                       )

        # Create the Loss Criterias 
        self.bce_criterion = nn.BCEWithLogitsLoss()
        self.ce_criterion = nn.CrossEntropyLoss()

        
        # Dynamic Weight Balancer 
        self.use_dynamic_loss = config.train_config.use_dynamic_loss
        self.balancer = Dynamic_Weight_Balancer(tasks=self.task_list,
                                                max_change_ratio=self.config.train_config.dynamic_max_change_ratio,
                                                update_every=self.config.train_config.dynamic_update_every
                                                ) if self.use_dynamic_loss else None

        # Amp and Scaler 
        self.use_amp = self.config.train_config.use_amp and self.device.startswith("cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Grad Clip
        self.grad_clip_norm = self.config.train_config.grad_clip_norm

    def _make_checkpoint_payload(
        self,
        epoch,
        best_epoch,
        train_multitask_loss,
        train_task_loss_dict,
        val_multitask_loss,
        val_task_loss_dict,
        val_task_acc_dict,
        best_val_multitask_loss,
        best_val_multitask_acc,
        save_condition
        ):


        checkpoint = {
            # training progress 
            "epoch": epoch,
            "best_epoch": best_epoch,
            "save_condition": save_condition,

            # model & optimizer (for resume) 
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.lr_scheduler.state_dict(),

            # config snapshot
            "config": vars(self.config) if hasattr(self.config, "__dict__") else self.config,

            # training metrics 
            "train_multitask_loss": train_multitask_loss,
            "train_task_loss_dict": train_task_loss_dict,

            # validation metrics
            "val_multitask_loss": val_multitask_loss,
            "val_task_loss_dict": val_task_loss_dict,
            "val_task_acc_dict": val_task_acc_dict,
            "best_val_multitask_loss": best_val_multitask_loss,
            "best_val_multitask_acc": best_val_multitask_acc,

            # scaler 
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None
        }

        return checkpoint

    def _save_checkpoint(self, payload, filename):

        # Ensure the directory exists
        os.makedirs(self.config.path_config.checkpoint_folder, exist_ok=True)

        save_path = os.path.join(self.config.path_config.checkpoint_folder, filename)

        # torch.save handles tensors, dicts, optimizer states, etc.
        torch.save(payload, save_path)

        # Optional logging
        if hasattr(self, "logger") and self.logger is not None:
            self.logger.info(f"Checkpoint saved: {save_path}")

        return save_path

    def load_checkpoint(self, checkpoint_path = "curr_epoch.pt"):
       
        checkpoint_path = os.path.join(self.config.path_config.checkpoint_folder, checkpoint_path)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location= self.device, weights_only=False)
        
        # Restore model state 
        if "model_state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint missing 'model_state_dict': {checkpoint_path}")
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Restore optimizer state
        if "optimizer_state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint missing 'optimizer_state_dict': {checkpoint_path}")
        
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore lr scheduler state
        if "scheduler_state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint missing 'scheduler_state_dict': {checkpoint_path}")
        
        self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load Scaler State 
        if self.use_amp and checkpoint.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Extract progress
        checkpoint_epoch = int(checkpoint.get("epoch", 1))
        start_epoch = int(checkpoint_epoch) + 1

        best_epoch = checkpoint.get("best_epoch", 1)
        best_val_loss = checkpoint.get("best_val_multitask_loss", float("inf"))
        best_val_acc = checkpoint.get("best_val_multitask_acc", float("-inf"))

        if hasattr(self, "logger") and self.logger:
            self.logger.info(f"Loaded Checkpoint from {checkpoint_path} at epoch {checkpoint_epoch}")

        return checkpoint, start_epoch, best_epoch, best_val_loss, best_val_acc

    def _to_device(self, batch: dict):

        """
        Moves all tensor values in the batch dict to the trainer device.
        Keeps non-tensors (ints, strings, etc.) unchanged.
        """
        out = {}
        for k, v in batch.items():
            out[k] = v.to(self.device) if torch.is_tensor(v) else v

        return out
    
    def _run_one_batch(self, task, raw_batch, mode):

        # Task Meta 
        meta = TASK_META_MAP[task]
        is_multilabel = bool(meta.get("multi_label", False))
        loss_type = meta["loss_type"] 
        read_type = meta["read_type"] 


        # For TAIL Tasks meta tells us where k comes from: "set_size" or "list_length"
        k_from = meta.get("k_from", None)
        pad_value = meta.get("pad_value", None) 

        # First we need to normalize the raw_batch
        batch = BATCH_ADAPTERS[task](batch = raw_batch)

        # The most common fields are stim (img_seq), resp (target), and seq_len (a tensor of sequence lengths for each stim in the batch)
        batch = self._to_device(batch)
        stim = batch["img_seq"]
        resp = batch["gt"]
        seq_len = batch["seq_len"]

        batch_size = stim.shape[0]

        # Next we perform the forward pass.
        # Model output --> logits_seq, mem_output, mem_h_n, projection_output, (cnn_output)
        # Logits seq shape --> (B, T, D) 
        # if model prepends a task token at the start, we need to drop that timestep 
        with torch.cuda.amp.autocast(enabled=(self.use_amp and mode == Modes.TRAIN)):
            logits_seq = forward_logits(stim = stim, 
                                        task = task, 
                                        seq_len = seq_len, 
                                        show_task_time = self.config.model_config.show_task_time,
                                        model = self.model)

            # We need to validate the logits shape to avoid shape bugs
            validate_logits(task = task, 
                            logits_seq = logits_seq, 
                            batch_size = batch_size)

            if read_type == ReadType.FINAL:
                logits_used, targets_used = select_final(
                    task = task,
                    logits_seq = logits_seq,
                    resp = resp,
                    seq_len = seq_len,
                    batch_size = batch_size,
                    is_multilabel = is_multilabel,
                    device = self.device
                )
                
                # Cast dtype based on target type
                logits_used, targets_used = cast_for_loss(loss_type = loss_type, 
                                                        logits_used = logits_used, 
                                                        targets_used = targets_used)

            elif read_type == ReadType.TAIL:
                logits_used, targets_used = select_tail(
                    task = task, 
                    logits_seq = logits_seq, 
                    resp = resp, 
                    seq_len = seq_len, 
                    batch = batch, 
                    batch_size = batch_size, 
                    k_from = k_from,
                    device = self.device
                )

                logits_used, targets_used = cast_for_loss(loss_type = loss_type, 
                                                        logits_used = logits_used, 
                                                        targets_used = targets_used)


            elif read_type == ReadType.SEQUENCE:
                logits_used, targets_used = select_seq(
                    task = task, 
                    logits_seq = logits_seq, 
                    resp = resp, 
                    seq_len = seq_len, 
                    batch_size = batch_size
                )

                # Mask out padding tokens if configured
                if pad_value is not None:
                    logits_used, targets_used = mask_padding(logits_used = logits_used, 
                                                            targets_used = targets_used, 
                                                            pad_value = pad_value)

                if loss_type == LossType.CATEGORICAL:
                    targets_used = targets_used.long()
                else:
                    targets_used = targets_used.float()

            else:
                raise ValueError(f"{task}: Unknown ReadType {read_type}")
                
            # Now its time to compute the loss
            loss = LOSS_FN_MAP[loss_type](self, logits = logits_used, targets = targets_used)
            
        # Time to compute metrics
        metrics = compute_metrics(mode = mode, 
                                  loss_type = loss_type, 
                                  logits_seq = logits_seq,
                                  batch = batch,
                                  batch_size = batch_size,
                                  task = task)
        
        metrics["loss"] = float(loss.detach().item())

        return loss, metrics

    def _run_one_epoch(self, mode, loaders, epoch_num=1):

        is_train_mode = (mode == Modes.TRAIN)

        if is_train_mode:
            self.model.train()

        else:
            self.model.eval()

        # Enable grads only in the train mode
        torch.set_grad_enabled(is_train_mode)

        # Using our task list we need to build an multitask batch 
        per_task_loaders = [loaders[t] for t in self.task_list]
        multitask_iterator = zip(*per_task_loaders)

        # Metrics 
        multitask_loss_sum = 0.0
        task_loss_sum = defaultdict(float) # sum of losses per task
        task_acc_sum = defaultdict(float) # sum of accuracies per task

        num_steps = 0

        detailed_acc = defaultdict(lambda: [0,0])

        num_batches = min(len(loader) for loader in per_task_loaders)

        # Loop over multi task batches
        for step_idx, multitask_batch in enumerate(tqdm(
            multitask_iterator,
            total=num_batches,
            leave = False,
            desc=f"{mode.name} | Epoch {epoch_num}")):

            if is_train_mode:
                # one optimizer step per multitask batch
                self.optimizer.zero_grad(set_to_none=True)

            # We will accumulate task losses into one multitask loss
            multitask_loss_this_step = 0.0

            # Task Loss Weight Balancing 
            task_loss_tensors = {}

            # Process each task batch inside this step
            for task_idx, task in enumerate(self.task_list):
                raw_task_batch = multitask_batch[task_idx]

                loss_tensor, metrics = self._run_one_batch(task=task, raw_batch=raw_task_batch, mode=mode)

                # Task Loss Weight Balancing 
                task_loss_tensors[task] = loss_tensor

                # update multitask_loss in this step
                # multitask_loss_this_step += loss_tensor
                task_loss_sum[task] += metrics["loss"]
                


                # record per task acc only for not train modes
                if not is_train_mode:
                    task_acc_sum[task] += float(metrics.get("acc", 0.0))
                
                if mode in [Modes.TEST, Modes.GEN_TEST] and "detailed" in metrics:

                    for sample in metrics["detailed"]:

                        metadata = sample["metadata"]
                        correct = sample["correct"]

                        # Custom Task Logic 
                        if "condition" in metadata:
                            key = f"{task.value}_{metadata['condition']}"
                            detailed_acc[key][0] += correct
                            detailed_acc[key][1] += 1

                            continue

                        for condition, value in metadata.items():
                            key = f"{task.value}_{condition}_{value}"
                            detailed_acc[key][0] += sample["correct"] # sample["correct"] is a float (0 or 1)
                            detailed_acc[key][1] += 1
                        
                        # if Specs.SET_SIZE.value in metadata and Specs.RETENTION_INTERVAL.value in metadata:
                        #     # e.g., "CD_Color_Task_RI_6_Set_Size_4"
                        #     comp_key = f"{task.value}_{Specs.RETENTION_INTERVAL.value}_{metadata[Specs.RETENTION_INTERVAL.value]}_{Specs.SET_SIZE.value}_{metadata[Specs.SET_SIZE.value]}"
                        #     detailed_acc[comp_key][0] += sample["correct"]
                        #     detailed_acc[comp_key][1] += 1
                            
                        # if Specs.LIST_LENGTH.value in metadata and Specs.DISTRACTOR_DIFF.value in metadata:
                        #     comp_key = f"{task.value}_{Specs.LIST_LENGTH.value}_{metadata[Specs.LIST_LENGTH.value]}_{Specs.DISTRACTOR_DIFF.value}_{metadata[Specs.DISTRACTOR_DIFF.value]}"
                        #     detailed_acc[comp_key][0] += sample["correct"]
                        #     detailed_acc[comp_key][1] += 1

                        parts = [task.value]
                        for k in sorted(metadata.keys()):
                            parts += [k, str(metadata[k])]
                        comp_key = "_".join(parts)
                        detailed_acc[comp_key][0] += correct
                        detailed_acc[comp_key][1] += 1

            # For Task Loss Weight Balancing 
            # Use the balancer to producde the final multitask loss for backprop
            if is_train_mode and self.balancer:
                multitask_loss_this_step = self.balancer.get_weighted_loss(task_loss_tensors)

            else:
                multitask_loss_this_step = sum(task_loss_tensors.values())
                
            

            # Backprop + Optimizer step
            # multitask_loss_this_step = multitask_loss_this_step / max(1, len(self.task_list))
            if is_train_mode:
                # multitask_loss_this_step.backward()
                # # Gradient clipping since we are planning to downscale the dataset size
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # self.optimizer.step()

                if self.use_amp:
                    self.scaler.scale(multitask_loss_this_step).backward()
                    # unscale before clipping
                    self.scaler.unscale_(self.optimizer)

                    if self.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                else:
                    multitask_loss_this_step.backward()
                    
                    if self.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                    
                    self.optimizer.step()

                
            # Track epoch level multitask loss
            # average accross tasks at the epoch level
            # divide this steps summed loss by number of tasks
            loss = float(multitask_loss_this_step.detach().item())
            # Changedthe form becuase of our weight balancing
            if self.use_dynamic_loss and self.balancer:
                multitask_loss_sum +=  loss / max(1e-8,sum(self.balancer.weights.values()))

            else:
                multitask_loss_sum += loss / len(self.task_list) 

            # multitask_loss_sum += multitask_loss_this_step.detach().item()
            num_steps += 1


        # Convert Sums to Avgs
        if num_steps == 0:
            epoch_multitask_loss_avg = 0.0
            epoch_task_loss_avg = {t: 0.0 for t in self.task_list}
            epoch_task_acc_avg = {t: 0.0 for t in self.task_list}

        else:
            epoch_multitask_loss_avg = multitask_loss_sum / num_steps
            epoch_task_loss_avg = {t: (task_loss_sum[t] / num_steps) for t in self.task_list}
            epoch_task_acc_avg = None

            if not is_train_mode:
                epoch_task_acc_avg = {t: (task_acc_sum[t] / num_steps) for t in self.task_list}

        if is_train_mode:
            return epoch_multitask_loss_avg, epoch_task_loss_avg, None, detailed_acc
        
        else:
            return epoch_multitask_loss_avg, epoch_task_loss_avg, epoch_task_acc_avg, detailed_acc

    def fit(self):

        train_loaders = {t: self.loaders[t]["train"] for t in self.task_list}
        val_loaders = {t: self.loaders[t]["val"] for t in self.task_list}


        # Initialise best tracking 
        best_epoch = 0
        best_val_loss = float("inf")
        best_val_acc = float("-inf")

        start_epoch = 1

        # Resume from previous checkpoint if needed
        if getattr(self.config.resumption_config, "resume", False):
            if self.config.resumption_config.resume_epoch not in [None, 0]:
                checkpoint, start_epoch, best_epoch, best_val_loss, best_val_acc = self.load_checkpoint(f"epoch_{str(self.config.resumption_config.resume_epoch).zfill(3)}.pt")
            else:
                checkpoint, start_epoch, best_epoch, best_val_loss, best_val_acc = self.load_checkpoint()

        # Epoch Loop
        for epoch in range(start_epoch, self.config.train_config.num_epochs + 1):

            if hasattr(self, "logger") and self.logger:
                self.logger.info("=" * 60)
                self.logger.info(f"Epoch {epoch} / {self.config.train_config.num_epochs}")

            # Train
            train_multitask_loss, train_task_loss_dict, _, _ = self._run_one_epoch(mode=Modes.TRAIN, loaders=train_loaders, epoch_num=epoch)

            # Val 
            val_multitask_loss, val_task_loss_dict, val_task_acc_dict, _ = self._run_one_epoch(mode=Modes.VAL, loaders=val_loaders, epoch_num=epoch)
            assert val_task_acc_dict != None

            # Task Loss Weight Balancing 
            # Update weights using the validation accuracies 
            if self.balancer:
                self.balancer.update_weights(val_task_acc_dict)

                if self.logger:
                    self.logger.info(f"Loss weights: " + ", ".join(
                        [f"{t.value}:{self.balancer.weights[t]:.3f}" for t in self.task_list]
                    ))

            if hasattr(self, "lr_scheduler") and self.lr_scheduler is not None:
                self.lr_scheduler.step(val_multitask_loss)


            # Compute an avg acc accross tasks 
            avg_val_acc = sum(val_task_acc_dict.values()) / max(1, len(val_task_acc_dict))

            # Decide if this is the best epoch
            # is_best = avg_val_acc > best_val_acc
            is_best = val_multitask_loss < best_val_loss
            if is_best:
                best_val_acc = avg_val_acc
                best_val_loss = val_multitask_loss
                best_epoch = epoch

            # Create Potential Checkpoint Payload
            cur_payload = self._make_checkpoint_payload(
                    epoch = epoch,
                    best_epoch = best_epoch,

                    train_multitask_loss = train_multitask_loss,
                    train_task_loss_dict = train_task_loss_dict,

                    val_multitask_loss = val_multitask_loss,
                    val_task_loss_dict = val_task_loss_dict,
                    val_task_acc_dict = val_task_acc_dict,

                    best_val_multitask_loss = best_val_loss,
                    best_val_multitask_acc = best_val_acc,
                    save_condition = f"{str(epoch).zfill(3)}"
                )
            
            # Save Checkpoints
            self._save_checkpoint(cur_payload, f"curr_epoch.pt")

            # Save checkpoints every {interval} epochs
            if epoch % self.config.train_config.test_interval == 0:
                self._save_checkpoint(cur_payload, f"epoch_{str(epoch).zfill(3)}.pt")
                            
            
            # Save best model 
            if is_best:
                best_payload = dict(cur_payload)
                best_payload["save_condition"] = "best_val_acc"
                self._save_checkpoint(best_payload, "best.pt")


            # Log
            if hasattr(self, "logger") and self.logger:
                self.logger.info(
                    f"Epoch {epoch} | "
                    f"Train Loss: {train_multitask_loss:.4f} | "
                    f"Val Loss: {val_multitask_loss:.4f} | "
                    f"Val Acc: {avg_val_acc:.4f} | "
                    f"Best Epoch: {best_epoch} Best ValLoss: {best_val_loss:.4f}"
                )

                print_log(self.logger, val_task_acc_dict, prefix="Val Acc")
                print_log(self.logger, val_task_loss_dict, prefix="Val Loss")

            # Save History to JSON
            epoch_metrics = {
                "epoch": epoch,
                "train_multitask_loss": float(train_multitask_loss),
                "train_task_loss": {t.value: float(v) for t, v in train_task_loss_dict.items()},
                "val_multitask_loss": float(val_multitask_loss),
                "val_task_loss": {t.value: float(v) for t, v in val_task_loss_dict.items()},
                "val_task_acc": {t.value: float(v) for t, v in val_task_acc_dict.items()}
            }

            history_file = os.path.join(self.config.path_config.output_folder, "train_history.json")

            if os.path.exists(history_file):
                with open(history_file, "r") as f:
                    history = json.load(f)
            else:
                history = []

            history.append(epoch_metrics)
            with open(history_file, "w") as f:
                json.dump(history, f)

            if self.logger:
                self.logger.info(f"Metrics for epoch {epoch} saved to {history_file}")


        if hasattr(self, "logger") and self.logger:
            self.logger.info(
                f"Training complete. Best epoch={best_epoch}, best_val_acc={best_val_acc:.4f}, best_val_loss={best_val_loss:.4f}"
            )

    def test(self, epoch, checkpoint_fname):

        # Loads the checkpoint
        checkpoint_path = os.path.join(self.config.path_config.checkpoint_folder, checkpoint_fname)
        self.load_checkpoint(checkpoint_fname)

        results = {
            "metadata": {
                "epoch": int(epoch) if epoch is not None else None,
                "checkpoint_path": checkpoint_path,
                "checkpoint_name": checkpoint_fname,
            },
        }

        # RUN TEST and GEN_TEST
        test_loaders = {task: self.loaders[task]["test"] for task in self.task_list}
        
        test_multitask_loss, test_task_loss, test_task_acc, test_detailed_acc = self._run_one_epoch(mode= Modes.TEST, loaders=test_loaders, epoch_num=epoch)
        assert test_task_acc != None

        test_avg_acc = calc_acc(test_task_acc)
        results["test"] = {
                "multitask_loss": float(test_multitask_loss),
                "avg_task_acc": float(test_avg_acc),
                "task_loss": {str(k): float(v) for k, v in test_task_loss.items()},
                "task_acc": {str(k): float(v) for k, v in test_task_acc.items()},
            }
        
        viz_results(
            epoch=epoch,
            detailed_acc=test_detailed_acc,
            config=self.config,
            task_list=self.task_list
        )
        
        if hasattr(self, "logger") and self.logger:
            self.logger.info("=== TEST RESULTS ===")
            self.logger.info(f"TEST | Multitask Loss: {test_multitask_loss:.4f} | Avg Task Acc: {test_avg_acc:.4f}")
            print_log(self.logger, test_task_acc, prefix="Test Acc")

        if self.config.execution_config.gen_test:
            gen_test_loaders = {task: self.loaders[task]["gen_test"] for task in self.task_list}
            gen_test_multitask_loss, gen_test_task_loss, gen_test_task_acc, gen_test_detailed_acc = self._run_one_epoch(mode= Modes.GEN_TEST, loaders=gen_test_loaders, epoch_num=epoch)
            gen_test_avg_acc = calc_acc(gen_test_task_acc)
            assert gen_test_task_acc != None

            results["gen_test"] = {
                "multitask_loss": float(gen_test_multitask_loss),
                "avg_task_acc": float(gen_test_avg_acc),
                "task_loss": {str(k): float(v) for k, v in gen_test_task_loss.items()},
                "task_acc": {str(k): float(v) for k, v in gen_test_task_acc.items()},
            }

            viz_results(
                epoch=epoch,
                detailed_acc=gen_test_detailed_acc,
                config=self.config,
                task_list=self.task_list
            )

            if hasattr(self, "logger") and self.logger:
                self.logger.info("=== GEN_TEST RESULTS ===")
                self.logger.info(f"TEST | Multitask Loss: {gen_test_multitask_loss:.4f} | Avg Task Acc: {gen_test_avg_acc:.4f}")
                print_log(self.logger, gen_test_task_acc, prefix="Gen Test Acc")

            return results, test_detailed_acc, gen_test_detailed_acc

        return results, test_detailed_acc, None








        


                