import torch
from src.utils.train_utils import Modes, LossType, ReadType, Specs, TASK_META_MAP, TASK_VIZ_SPECS, mask_padding
from src.tasks import Tasks


def compute_metrics(mode, loss_type, logits_seq, batch, batch_size, task):
    
    metrics = {}
    if mode == Modes.TRAIN:
        return metrics

    with torch.no_grad():
        meta = TASK_META_MAP[task]

        read_type = meta["read_type"]
        pad_value = meta.get("pad_value", None)

        # convert the entire batch to predictions first, shape --> (B,T, ..)
        if loss_type == LossType.BINARY:
            full_pred = (torch.sigmoid(logits_seq) >= 0.5).float()
        else:
            full_pred = torch.argmax(logits_seq, dim = -1)

        batch_results = []
        epoch_acc_sum = 0
        total_epoch_steps = 0

        for i in range(batch_size):
            seq_len = int(batch["seq_len"][i].item())

            if read_type == ReadType.FINAL:
                idx = seq_len - 1
                sample_pred = full_pred[i, idx].unsqueeze(0)
                sample_gt = batch["gt"][i, idx].unsqueeze(0)

            elif read_type == ReadType.TAIL:
                k = int(batch[meta["k_from"].value][i].item())
                sample_pred = full_pred[i, seq_len-k : seq_len]
                sample_gt = batch["gt"][i, seq_len-k : seq_len]

            else: # SEQUENCE
                sample_pred = full_pred[i, :seq_len]
                sample_gt = batch["gt"][i, :seq_len]

                if pad_value is not None:
                    sample_pred, sample_gt = mask_padding(logits_used= sample_pred, targets_used = sample_gt, pad_value = pad_value)
            
            # Ensure shapes batch (N, 1)
            sample_pred = sample_pred.reshape(-1)
            sample_gt = sample_gt.reshape(-1)

            epoch_acc_sum += (sample_pred == sample_gt).float().sum().item()
            total_epoch_steps += sample_gt.size(0)

            if mode in [Modes.TEST, Modes.GEN_TEST]:
                
                # Custom Logic for VSR and VSRec
                if task in [Tasks.VISUAL_SERIAL_RECALL, Tasks.VISUAL_SERIAL_RECOGNITION]:
                    batch_results.append({
                        "correct": float(torch.equal(sample_pred, sample_gt.float() if loss_type == LossType.BINARY else sample_gt)),
                        "metadata": {"condition": "Whole Sequence Correct"}
                    })

                # METADATA & SERIAL POSITION
                res_metadata = {}
                for spec_key in TASK_VIZ_SPECS.get(task, []):
                    if spec_key == Specs.SERIAL_POSITION: continue
                    
                    # Custom logic for Spatial Free Recall
                    if task == Tasks.SPATIAL_FREE_RECALL and spec_key == Specs.LIST_LENGTH:
                        res_metadata[spec_key.value] = seq_len - 1
                    else:
                        val = batch.get(spec_key.value)
                        if val is not None:
                            res_metadata[spec_key.value] = val[i].item()

                if task != Tasks.SPATIAL_FREE_RECALL:
                    for step_idx in range(sample_gt.size(0)):
                        step_metadata = res_metadata.copy()
                        
                        if Specs.SERIAL_POSITION in TASK_VIZ_SPECS.get(task, []):
                            if task == Tasks.VISUAL_ITEM_RECOGNITION:
                                step_metadata[Specs.SERIAL_POSITION.value] = int(batch["gt_index"][i].item() + 1)
                            else:
                                step_metadata[Specs.SERIAL_POSITION.value] = step_idx + 1
                        
                        batch_results.append({
                            "correct": float(sample_pred[step_idx] == sample_gt[step_idx]),
                            "metadata": step_metadata
                        })

                # Custom Logic For Spatial Free Recall
                if task == Tasks.SPATIAL_FREE_RECALL:
                    sfr_res = compute_sfr_metrics(
                        logits=logits_seq[i, seq_len-1], 
                        recall_gt=batch["recall_gt_original"][i],
                        list_length=seq_len - 1
                    )
                    for key, correct in sfr_res.items():
                        batch_results.append({
                            "correct": float(correct),
                            "metadata": {"condition": key.title()}
                        })

        acc = epoch_acc_sum / max(1, total_epoch_steps)

        if task == Tasks.SPATIAL_FREE_RECALL:
            acc = acc / 100.0

        metrics["acc"] = acc

        if mode in [Modes.TEST, Modes.GEN_TEST]:
            metrics["detailed"] = batch_results  

    return metrics          



def compute_sfr_metrics(logits, recall_gt, list_length):

    # First we need to get the top K predictions from the model
    _, pred_idxs = torch.topk(logits, k = list_length, sorted = True)
    pred_idxs = pred_idxs.cpu()
    valid_gt = recall_gt[recall_gt != -1].cpu()

    # Now we check if the forward order matches 
    forward_order = torch.equal(pred_idxs, valid_gt)

    # Now we perform the no order check
    no_order = torch.equal(torch.sort(pred_idxs)[0], torch.sort(valid_gt)[0])

    return {
        "Forward Order": forward_order,
        "No Order": no_order,
    }

def calc_acc(task_acc_dict):
    avg_acc = sum(task_acc_dict.values()) / max(1, len(task_acc_dict))
    return avg_acc



