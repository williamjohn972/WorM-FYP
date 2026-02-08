from enum import Enum
import torch
from src.tasks import Tasks
from src.model import Show_Task

class Modes(Enum):

    TRAIN = "train"   
    VAL = "val"
    TEST = "test"
    GEN_TEST = "gen_test"

class ReadType(Enum):
    FINAL = "final"        # use last relevant timestep only 
    TAIL = "tail"          # use the last k steps of (list_length or set_size)
    SEQUENCE = "sequence"  # use all the steps up to seq_len and ignore the padding

class LossType(Enum):
    BINARY = "binary"
    CATEGORICAL = "categorical"

class Specs(Enum):
    SYMETRY_OFFSET = "symetric_offset"
    PART_SIZE = "part_size"
    SET_SIZE = "set_size"
    SERIAL_POSITION = "serial_position"
    RETENTION_INTERVAL = "retention_interval"
    DISTRACTOR_DIFF = "distractor_diff"
    LIST_LENGTH = "list_length"
    CONJ_GT = "conj_label"
    

TASK_META_MAP = {
    Tasks.SPATIAL_COORDINATION: dict(
        loss_type = LossType.BINARY,
        read_type = ReadType.FINAL,
        k_from = None,
        pad_value = None,
        multi_label = False,
    ),

    Tasks.SPATIAL_FREE_RECALL: dict(
        loss_type = LossType.BINARY,
        read_type = ReadType.FINAL,
        multi_label = True,
        k_from = None,
        pad_value = None
    ),

    Tasks.SPATIAL_INTEGRATION: dict(
        loss_type = LossType.BINARY,
        read_type = ReadType.FINAL,
        k_from = None,
        pad_value = None,
        multi_label = False,
    ),

    Tasks.SPATIAL_MEMORY_UPDATING: dict(
        loss_type = LossType.CATEGORICAL,
        read_type = ReadType.TAIL,
        k_from = Specs.SET_SIZE,
        pad_value = None,
        multi_label = False,
    ),

    Tasks.SPATIAL_TASK_SWITCHING: dict(
        loss_type = LossType.CATEGORICAL,
        read_type = ReadType.SEQUENCE,
        k_from = None,
        pad_value = 2,
        multi_label = False,
    ), 

    Tasks.VISUAL_ITEM_RECOGNITION: dict(
        loss_type = LossType.BINARY,
        read_type = ReadType.FINAL,
        k_from = None,
        pad_value = None,
        multi_label = False,
    ),

    Tasks.VISUAL_SERIAL_RECALL: dict(
        loss_type = LossType.CATEGORICAL,
        read_type = ReadType.TAIL,
        k_from = Specs.LIST_LENGTH,
        pad_value = None,
        multi_label = False,
    ),

    Tasks.VISUAL_SERIAL_RECOGNITION: dict(
        loss_type = LossType.BINARY,
        read_type = ReadType.TAIL,
        k_from = Specs.LIST_LENGTH,
        pad_value = None,
        multi_label = False,
    ),

    Tasks.CHANGE_DETECTION_COLOR: dict(
        loss_type = LossType.BINARY,
        read_type = ReadType.FINAL,
        k_from = None,
        pad_value = None,
        multi_label = False,
    ),

    Tasks.CHANGE_DETECTION_ORIENTATION: dict(
        loss_type = LossType.BINARY,
        read_type = ReadType.FINAL,
        k_from = None,
        pad_value = None,
        multi_label = False,
    ),

    Tasks.CHANGE_DETECTION_SIZE: dict(
        loss_type = LossType.BINARY,
        read_type = ReadType.FINAL,
        k_from = None,
        pad_value = None,
        multi_label = False,
    ),

    Tasks.CHANGE_DETECTION_GAP: dict(
        loss_type = LossType.BINARY,
        read_type = ReadType.FINAL,
        k_from = None,
        pad_value = None,
        multi_label = False,
    ),

    Tasks.CHANGE_DETECTION_CONJ: dict(
        loss_type = LossType.BINARY,
        read_type = ReadType.FINAL,
        k_from = None,
        pad_value = None,
        multi_label = False,
    ),
}

BATCH_ADAPTERS = {
    Tasks.SPATIAL_COORDINATION: lambda batch: dict(img_seq = batch[0], gt = batch[1], seq_len = batch[2], symetry_offset = batch[3]),
    Tasks.SPATIAL_FREE_RECALL: lambda batch: dict(img_seq = batch[0], gt = batch[1], # gt one hot  
                                                  seq_len = batch[2], recall_gt_original=batch[3]),
    Tasks.SPATIAL_INTEGRATION: lambda batch: dict(img_seq = batch[0], gt = batch[1], seq_len = batch[2], part_size = batch[3]),
    Tasks.SPATIAL_MEMORY_UPDATING: lambda batch: dict(img_seq = batch[0], gt = batch[1], seq_len = batch[2], set_size = batch[3]),
    Tasks.SPATIAL_TASK_SWITCHING: lambda batch: dict(img_seq = batch[0], gt = batch[1], seq_len = batch[2]), 

    Tasks.VISUAL_ITEM_RECOGNITION: lambda batch: dict(img_seq = batch[0], gt = batch[1], seq_len = batch[2], retention_interval = batch[3], gt_index = batch[4]),
    Tasks.VISUAL_SERIAL_RECALL: lambda batch: dict(img_seq = batch[0], gt = batch[1], seq_len = batch[2], list_length = batch[3]),
    Tasks.VISUAL_SERIAL_RECOGNITION: lambda batch: dict(img_seq = batch[0], gt = batch[1], seq_len = batch[2], list_length = batch[3], distractor_diff = batch[4]),

    Tasks.CHANGE_DETECTION_COLOR: lambda batch: dict(img_seq = batch[0], gt = batch[1], seq_len = batch[2], retention_interval = batch[3], set_size = batch[4]),
    Tasks.CHANGE_DETECTION_ORIENTATION: lambda batch: dict(img_seq = batch[0], gt = batch[1], seq_len = batch[2], retention_interval = batch[3], set_size = batch[4]),
    Tasks.CHANGE_DETECTION_SIZE: lambda batch: dict(img_seq = batch[0], gt = batch[1], seq_len = batch[2], retention_interval = batch[3], set_size = batch[4]),
    Tasks.CHANGE_DETECTION_GAP: lambda batch: dict(img_seq = batch[0], gt = batch[1], seq_len = batch[2], retention_interval = batch[3], set_size = batch[4]),
    Tasks.CHANGE_DETECTION_CONJ: lambda batch: dict(img_seq = batch[0], gt = batch[1], seq_len = batch[2], retention_interval = batch[3], set_size = batch[4], conj_label = batch[5]),
}

LOSS_FN_MAP = {
    LossType.BINARY: lambda self, logits, targets: self.bce_criterion(logits, targets),
    LossType.CATEGORICAL: lambda self, logits, targets: self.ce_criterion(logits, targets)
}

PRED_FN_MAP = {
  LossType.BINARY: lambda logits: (torch.sigmoid(logits) >= 0.5),
  LossType.CATEGORICAL: lambda logits: torch.argmax(logits, dim=-1),
}

ACC_FN_MAP = {
  LossType.BINARY: lambda pred, targets: (pred == targets).float().mean().item(),
  LossType.CATEGORICAL: lambda pred, targets: (pred == targets).float().mean().item(),
}

READ_SELECTOR_MAP = {
  ReadType.FINAL: lambda self, task, logits_seq, batch, meta: self._select_final(),
  ReadType.TAIL:  lambda self, task, logits_seq, batch, meta: self._select_tail(),
  ReadType.SEQUENCE: lambda self, task, logits_seq, batch, meta: self._select_seq(),
}



def pick_resp_window(resp_tensor, batch_index, start_t, end_t, task):
    """
    A helper function that slices resp by time only when resp is time-indexed
    """
    if resp_tensor.ndim == 2: # resp shape (B, T)
        return resp_tensor[batch_index, start_t:end_t]
    
    if resp_tensor.ndim == 3 and resp_tensor.shape[-1] == 1: # resp shape (B, T, 1)
        return resp_tensor[batch_index, start_t:end_t, 0]
    
    raise ValueError(f"{task}: time-window pick requested, but resp has shape {tuple(resp_tensor.shape)}")

def select_final(task, logits_seq, resp, seq_len, batch_size, is_multilabel, device):
    # shape --> (B,)
    idx = (seq_len - 1).clamp_min(0).long()

    # logits_used  -> (B, D)
    logits_used = logits_seq[torch.arange(batch_size, device= device), idx, :]

    # targets 
    if is_multilabel:
        # targets_used -> (B, D) (D=100)
        targets_used = resp.float()

        if logits_used.ndim != 2 or targets_used.ndim != 2:
            raise RuntimeError(f"{task}: multilabel expects logits (B,D) and targets (B,D)."
                                f"got logits {tuple(logits_used.shape)}, targets {tuple(targets_used.shape)}")
        
        if logits_used.shape[0] != targets_used.shape[0] or logits_used.shape[1] != targets_used.shape[1]:
            raise RuntimeError(f"{task}: multilabel shape mismatch. "
                            f"logits {tuple(logits_used.shape)} vs targets {tuple(targets_used.shape)}")

    else:
        # Non mulilabel FINAL tasks
        # resp should be time indexed (B,T) or (B,T,1)
        # we pick target at the same timestep index
        if resp.ndim == 2:
            targets_used = resp[torch.arange(batch_size, device= device), idx]

        elif resp.ndim == 3 and resp.shape[-1] == 1:
            targets_used = resp[torch.arange(batch_size, device= device), idx, 0]

        elif resp.ndim == 1 and resp.shape[0] == batch_size:
            targets_used = resp

        else:
            raise ValueError(f"{task}: FINAL expects resp shaped (B,T) or (B,T,1) or (B,), got {tuple(resp.shape)}")
        
    return logits_used, targets_used
    
def select_tail(task, logits_seq, resp, seq_len, batch, batch_size, k_from, device):
        # use the last k steps of each sequence
        # k comes from the set_size or the list_length

        if k_from is None:
            raise ValueError(f"{task}: TAIL requires meta['k_from'] to be set (e.g., 'set_size' or 'list_length').")
        
        k_per_sample = batch[k_from.value].to(device).long() # shape --> (B,)

        logits_chunks = []
        targets_chunks = []

        # for each sample in the batch
        for i in range(batch_size):
            end = int(seq_len[i].item())
            k = int(k_per_sample[i].item())
            start = max(0, end - k)

            # logits window: (k, D)
            logits_chunks.append(logits_seq[i, start:end, :])

            # targets_window: (k, )
            targets_chunks.append(pick_resp_window(resp_tensor = resp, batch_index = i, start_t = start, end_t = end, task = task))

        
        # Concatenate accross batch -> (N,D) and (N, )
        logits_used = torch.cat(logits_chunks, dim=0)
        targets_used = torch.cat(targets_chunks, dim=0)


        return logits_used, targets_used

def select_seq(task, logits_seq, resp, seq_len, batch_size):
    # use all steps up to seq_len for each sample (ignore padding beyond seq_len)
    logits_chunks = []
    targets_chunks = []

    # for each sample in the batch
    for i in range(batch_size):
        end = int(seq_len[i].item())
        logits_chunks.append(logits_seq[i, :end, :])
        targets_chunks.append(pick_resp_window(resp_tensor = resp, batch_index = i, start_t = 0, end_t = end, task = task))


    logits_used = torch.cat(logits_chunks, dim=0) # (N,D)
    targets_used = torch.cat(targets_chunks, dim=0) # (N,)
    

    return logits_used, targets_used
    
def mask_padding(logits_used: torch.Tensor, targets_used: torch.Tensor, pad_value: int):
    """
    Applies pad masking for SEQUENCE tasks (e.g., STS pad token=2).
    """
    
    valid_mask = targets_used != pad_value
    logits_used = logits_used[valid_mask]
    targets_used = targets_used[valid_mask]

    return logits_used, targets_used

def cast_for_loss(loss_type, logits_used: torch.Tensor, targets_used:torch.Tensor):
    """
    Ensures dtypes/shapes match the loss:
    - BCEWithLogitsLoss expects float targets, logits can be (N,) or (N,1)
    - CrossEntropyLoss expects long targets and logits (N, C)
    """

    if loss_type == LossType.BINARY:
    # If logits are (N,1), squeeze to (N,)
        if logits_used.ndim == 2 and logits_used.shape[-1] == 1:
            logits_used = logits_used[:, 0]
        targets_used = targets_used.float()
    else:
        targets_used = targets_used.long()

    return logits_used, targets_used

def forward_logits(stim:torch.Tensor, task, seq_len: torch.Tensor, model, show_task_time):
    """
    Runs the model forward pass and returns logits_seq with shape (B, T, D).
    Applies Show_Task.START trimming if needed.
    """

    model_out = model(stim, task, seq_len)
    logits_seq = model_out[0]

    if show_task_time == Show_Task.START:
        logits_seq = logits_seq[:, 1:, :]

    return logits_seq

def validate_logits(task, logits_seq: torch.Tensor, batch_size: int):
    """
    Debug-friendly check for the expected logits shape.
    """

    if logits_seq.ndim != 3 or logits_seq.shape[0] != batch_size:
        raise RuntimeError(f"{task}: expected logits_seq shape (B,T,D), got {tuple(logits_seq.shape)}")

