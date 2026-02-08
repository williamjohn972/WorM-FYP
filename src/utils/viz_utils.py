import os
from typing import Dict, List, Tuple

from matplotlib import pyplot as plt

from src.tasks import Tasks
from src.config import Config
from src.utils.train_utils import Specs


TASK_VIZ_MAP = {
    # Spatial tasks
    Tasks.SPATIAL_COORDINATION: lambda task_data, save_folder: viz_spatial_coordination(task_data, save_folder),
    Tasks.SPATIAL_FREE_RECALL: lambda task_data, save_folder: viz_spatial_free_recall(task_data, save_folder),
    Tasks.SPATIAL_INTEGRATION: lambda task_data, save_folder: viz_spatial_integration(task_data, save_folder),
    Tasks.SPATIAL_MEMORY_UPDATING: lambda task_data, save_folder: viz_spatial_memory_updating(task_data, save_folder),
    Tasks.SPATIAL_TASK_SWITCHING: None,

    # Visual tasks
    Tasks.VISUAL_ITEM_RECOGNITION: lambda task_data, save_folder: viz_visual_item_recognition(task_data, save_folder),
    Tasks.VISUAL_SERIAL_RECALL: lambda task_data, save_folder: viz_visual_serial_recall(task_data, save_folder),
    Tasks.VISUAL_SERIAL_RECOGNITION: lambda task_data, save_folder: viz_visual_serial_recognition(task_data, save_folder),

    # Change detection tasks
    Tasks.CHANGE_DETECTION_COLOR: lambda task_data, save_folder: viz_change_detection(task_data, save_folder),
    Tasks.CHANGE_DETECTION_ORIENTATION: lambda task_data, save_folder: viz_change_detection(task_data, save_folder),
    Tasks.CHANGE_DETECTION_SIZE: lambda task_data, save_folder: viz_change_detection(task_data, save_folder),
    Tasks.CHANGE_DETECTION_GAP: lambda task_data, save_folder: viz_change_detection(task_data, save_folder),
    Tasks.CHANGE_DETECTION_CONJ: lambda task_data, save_folder: viz_change_detection(task_data, save_folder),
}


def viz_results(epoch, detailed_acc, config: Config, task_list):
    """
    Save task-specific behavioral plots for a given epoch.

    detailed_acc maps: key -> [num_correct, num_total]
    """
    epoch_folder = os.path.join(config.path_config.output_folder, f"epoch_{str(epoch).zfill(3)}")
    os.makedirs(epoch_folder, exist_ok=True)

    for task in task_list:
        task_key_prefix = task.value  
        task_detailed_data = {k: v for k, v in detailed_acc.items() if task_key_prefix in k}

        if not task_detailed_data:
            continue

        viz_func = TASK_VIZ_MAP.get(task, None)

        if viz_func:
            viz_func(task_detailed_data, epoch_folder)



def viz_spatial_coordination(task_data, save_folder):
    """
     Accuracy vs Set Size  
     Accuracy vs Symmetry Offset
    """

    task_name = Tasks.SPATIAL_COORDINATION.value
    _plot_scalar_accuracy(
        task_data=task_data,
        spec=Specs.SET_SIZE,
        save_folder=save_folder,
        title=f"{task_name.title()}: {Specs.SET_SIZE.value} vs Accuracy",
        filename=f"{task_name.title()}_{Specs.SET_SIZE.value.title()}_Accuracy.png",
        y_label="Accuracy",
    )

    _plot_scalar_accuracy(
        task_data=task_data,
        spec=Specs.SYMETRY_OFFSET,
        save_folder=save_folder,
        title=f"{task_name.title()}: {Specs.SYMETRY_OFFSET.value} vs Accuracy",
        filename=f"{task_name.title()}_{Specs.SYMETRY_OFFSET.value.title()}_Accuracy.png",
        y_label="Accuracy",
    )


def viz_spatial_free_recall(task_data, save_folder):
    """
      List_length vs Serial_Position  
      Categorical bar chart (Forward Order / No Order / etc.)
    """

    _plot_serial_position_by_primary_spec(
        task=Tasks.SPATIAL_FREE_RECALL,
        task_data=task_data,
        primary_spec=Specs.LIST_LENGTH,
        save_folder=save_folder,
        output_filename=f"{Tasks.SPATIAL_FREE_RECALL.value.title()}_{Specs.LIST_LENGTH.value.title()}_Serial_Position.png",
    )

    _plot_free_recall_strategy_bars(
        task=Tasks.SPATIAL_FREE_RECALL,
        task_data=task_data,
        save_folder=save_folder,
        output_filename=f"{Tasks.SPATIAL_FREE_RECALL.value.title()}_Strategy_Comparison.png",
    )


def viz_spatial_integration(task_data, save_folder):
    """
     Number of Integrations vs Accuracy  
     Part Size vs Accuracy
    """
    task_name = Tasks.SPATIAL_INTEGRATION.value

    # Part size points
    part_size_points = _collect_scalar_points(task_data, Specs.PART_SIZE)
    if not part_size_points:
        return

    # Number of integrations (derived from part size)
    integrations_points = []
    for part_size, accuracy in part_size_points:

        # num_integrations = (12 // part_size) - 1
        num_integrations = (12 // part_size) - 1
        integrations_points.append((num_integrations, accuracy))

    _plot_line(
        points=_dedupe_and_sort_points(integrations_points),
        title=f"{task_name.title()}: Complexity vs Accuracy",
        x_label="Number of Integrations",
        y_label="Accuracy",
        save_path=os.path.join(save_folder, f"{task_name.title()}_Num_Integration_Accuracy.png"),
    )

    # Part size vs accuracy
    _plot_line(
        points=_dedupe_and_sort_points(part_size_points),
        title=f"{task_name.title()}: Part Size Accuracy",
        x_label="Part Size",
        y_label="Accuracy",
        save_path=os.path.join(save_folder, f"{task_name.title()}_{Specs.PART_SIZE.value.title()}_Accuracy.png"),
    )


def viz_spatial_memory_updating(task_data, save_folder):
    """
      Serial position curves vs Set Size  
      Collapsed set size accuracy (mean over serial positions)
    """
    task = Tasks.SPATIAL_MEMORY_UPDATING

    _plot_serial_position_by_primary_spec(
        task=task,
        task_data=task_data,
        primary_spec=Specs.SET_SIZE,
        save_folder=save_folder,
        output_filename=f"{task.value.title()}_{Specs.SET_SIZE.value.title()}_Serial_Position.png",
    )

    _plot_accuracy_collapsed_over_serial_position(
        task=task,
        task_data=task_data,
        primary_spec=Specs.SET_SIZE,
        save_folder=save_folder,
        output_filename=f"{task.value.title()}_{Specs.SET_SIZE.value.title()}_Accuracy.png",
    )


def viz_visual_item_recognition(task_data, save_folder) -> None:
    """
      Set Size vs Retention Interval curves  
      Serial position curves by retention interval  
      Collapsed retention interval accuracy (mean over serial positions)
    """

    task = Tasks.VISUAL_ITEM_RECOGNITION

    _plot_set_size_vs_retention_interval_interaction(
        task=task,
        task_data=task_data,
        save_folder=save_folder,
        output_filename=f"{task.value.title()}_{Specs.RETENTION_INTERVAL.value.title()}_{Specs.SET_SIZE.value.title()}_Accuracy.png",
    )

    _plot_serial_position_by_primary_spec(
        task=task,
        task_data=task_data,
        primary_spec=Specs.RETENTION_INTERVAL,
        save_folder=save_folder,
        output_filename=f"{task.value.title()}_{Specs.RETENTION_INTERVAL.value.title()}_Serial_Position.png",
    )

    _plot_accuracy_collapsed_over_serial_position(
        task=task,
        task_data=task_data,
        primary_spec=Specs.RETENTION_INTERVAL,
        save_folder=save_folder,
        output_filename=f"{task.value.title()}_{Specs.RETENTION_INTERVAL.value.title()}_Accuracy.png",
    )


def viz_visual_serial_recall(task_data, save_folder):
    """
    Serial position curves by list length.
    """
    task = Tasks.VISUAL_SERIAL_RECALL
    _plot_serial_position_by_primary_spec(
        task=task,
        task_data=task_data,
        primary_spec=Specs.LIST_LENGTH,
        save_folder=save_folder,
        output_filename=f"{task.value.title()}_{Specs.LIST_LENGTH.value.title()}_Serial_Position.png",
    )


def viz_visual_serial_recognition(task_data, save_folder) -> None:
    """
      List Length x Serial Position, collapsed over distractor  
      Accuracy per distractor diff
    """
    task = Tasks.VISUAL_SERIAL_RECOGNITION

    _plot_vsrec_list_length_by_serial_position_collapsed_over_distractor(
        task_data=task_data,
        save_folder=save_folder,
        output_filename="VSRec_Task_Accuracy.png",
    )

    _plot_vsrec_overall_accuracy_by_distractor(
        task_data=task_data,
        save_folder=save_folder,
        output_filename="VSRec_Task_Distractor_Accuracy_Overall.png",
    )


def viz_change_detection(task_data, save_folder) -> None:
    """
    Set Size vs Retention Interval curves
    """

    example_key = next(iter(task_data.keys()))
    task_name = example_key.split("_")[0] 

    output_file_name = f"{task_name.title()}_{Specs.RETENTION_INTERVAL.value.title()}_{Specs.SET_SIZE.value.title()}_Accuracy.png"
    
    _plot_set_size_vs_retention_interval_interaction(
        task=None,
        task_data=task_data,
        save_folder=save_folder,
        output_filename=output_file_name,  
    )


def _plot_serial_position_by_primary_spec(
    task: Tasks,
    task_data,
    primary_spec: Specs,
    save_folder,
    output_filename,
):
    """
    Plot Serial Position (x) vs Accuracy (y), with one line per primary_spec value.
    Assumes keys contain both:
        _{primary_spec}_<int> and _serial_position_<int>
    """
    primary_spec_tag = f"_{primary_spec.value}_"
    serial_position_tag = f"_{Specs.SERIAL_POSITION.value}_"

    primary_values = _sorted_unique_ints_from_keys(task_data.keys(), primary_spec_tag)
    if not primary_values:
        return

    plt.figure(figsize=(8, 6))

    for primary_value in primary_values:
        # collect (position, accuracy) for this primary value
        position_accuracy_points = []
        for key, (num_correct, num_total) in task_data.items():
            if f"{primary_spec_tag}{primary_value}" not in key:
                continue
            if serial_position_tag not in key:
                continue
            if num_total <= 0:
                continue

            serial_position_value = _extract_int_from_key(key, serial_position_tag)
            if serial_position_value is None:
                continue

            accuracy = num_correct / num_total
            position_accuracy_points.append((serial_position_value, accuracy))

        if not position_accuracy_points:
            continue

        position_accuracy_points.sort()
        x_positions, y_accuracies = zip(*position_accuracy_points)
        plt.plot(x_positions, y_accuracies, marker="o", label=f"{primary_spec.value}: {primary_value}")

    plt.title(f"{task.value.title()} Serial Position Analysis")
    plt.xlabel("Serial Position")
    plt.ylabel("Accuracy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, output_filename))
    plt.close()


def _plot_accuracy_collapsed_over_serial_position(
    task: Tasks,
    task_data,
    primary_spec: Specs,
    save_folder,
    output_filename,
):
    """
    Collapse over serial position:
      For each primary_spec value, compute mean accuracy across serial positions.
    """
    primary_spec_tag = f"_{primary_spec.value}_"
    serial_position_tag = f"_{Specs.SERIAL_POSITION.value}_"

    accuracy_by_primary: Dict[int, List[float]] = {}

    for key, (num_correct, num_total) in task_data.items():
        if primary_spec_tag not in key or serial_position_tag not in key:
            continue
        if num_total <= 0:
            continue

        primary_value = _extract_int_from_key(key, primary_spec_tag)
        serial_position_value = _extract_int_from_key(key, serial_position_tag)
        if primary_value is None or serial_position_value is None:
            continue

        accuracy_by_primary.setdefault(primary_value, []).append(num_correct / num_total)

    if not accuracy_by_primary:
        return

    primary_values_sorted = sorted(accuracy_by_primary.keys())
    mean_accuracies = [
        sum(accuracy_by_primary[p]) / len(accuracy_by_primary[p])
        for p in primary_values_sorted
    ]

    plt.figure(figsize=(8, 6))
    plt.plot(primary_values_sorted, mean_accuracies, marker="o")
    plt.title(f"{task.value.title()} {primary_spec.value.title()} Accuracy (collapsed over Serial Position)")
    plt.xlabel(primary_spec.value.title())
    plt.ylabel("Accuracy")
    plt.xticks(primary_values_sorted)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, output_filename))
    plt.close()


def _plot_set_size_vs_retention_interval_interaction(
    task,
    task_data,
    save_folder,
    output_filename,
):
    """
    Plot Set Size (x) vs Accuracy (y), one curve per Retention Interval.
    """
    retention_interval_tag = f"_{Specs.RETENTION_INTERVAL.value}_"
    set_size_tag = f"_{Specs.SET_SIZE.value}_"

    retention_intervals = _sorted_unique_ints_from_keys(task_data.keys(), retention_interval_tag)
    if not retention_intervals:
        return

    # Determine filename
    if output_filename is None:
        inferred_task_label = task.value.title() if task is not None else "Task"
        output_filename = f"{inferred_task_label}_{Specs.RETENTION_INTERVAL.value.title()}_{Specs.SET_SIZE.value.title()}_Accuracy.png"

    # (ri, set_size) -> [correct_sum, total_sum]
    aggregated_counts: Dict[Tuple[int, int], List[float]] = {}

    for key, (num_correct, num_total) in task_data.items():
        if retention_interval_tag not in key or set_size_tag not in key:
            continue
        if num_total <= 0:
            continue

        ri_value = _extract_int_from_key(key, retention_interval_tag)
        set_size_value = _extract_int_from_key(key, set_size_tag)
        if ri_value is None or set_size_value is None:
            continue

        aggregated_counts.setdefault((ri_value, set_size_value), [0.0, 0.0])
        aggregated_counts[(ri_value, set_size_value)][0] += num_correct
        aggregated_counts[(ri_value, set_size_value)][1] += num_total

    if not aggregated_counts:
        return

    plt.figure(figsize=(8, 6))

    for ri in retention_intervals:
        set_size_points = []
        for (ri_value, set_size_value), (correct_sum, total_sum) in aggregated_counts.items():
            if ri_value != ri or total_sum <= 0:
                continue
            set_size_points.append((set_size_value, correct_sum / total_sum))

        if not set_size_points:
            continue

        set_size_points.sort()
        x_set_sizes, y_accuracies = zip(*set_size_points)
        plt.plot(x_set_sizes, y_accuracies, marker="s", label=f"Retention Interval: {ri}s")

    plot_title = (task.value.title() if task is not None else "Task") + " Set Size vs Retention Interval"
    plt.title(plot_title)
    plt.xlabel("Set Size")
    plt.ylabel("Accuracy")
    plt.legend(title="Retention Interval", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, output_filename))
    plt.close()



def _plot_vsrec_overall_accuracy_by_distractor(
    task_data,
    save_folder,
    output_filename,
):
    """
    VSRec overall accuracy vs distractor difference.
    Aggregates across list length and serial position via sum(correct)/sum(total).
    """
    distractor_tag = f"_{Specs.DISTRACTOR_DIFF.value}_"
    distractor_totals: Dict[int, List[float]] = {}  # d -> [correct_sum, total_sum]

    for key, (num_correct, num_total) in task_data.items():
        if distractor_tag not in key:
            continue
        distractor_value = _extract_int_from_key(key, distractor_tag)
        if distractor_value is None:
            continue

        distractor_totals.setdefault(distractor_value, [0.0, 0.0])
        distractor_totals[distractor_value][0] += num_correct
        distractor_totals[distractor_value][1] += num_total

    if not distractor_totals:
        return

    distractor_values_sorted = sorted(distractor_totals.keys())
    accuracies = [
        distractor_totals[d][0] / distractor_totals[d][1]
        for d in distractor_values_sorted
        if distractor_totals[d][1] > 0
    ]

    plt.figure(figsize=(8, 6))
    plt.plot(distractor_values_sorted, accuracies, marker="o")
    plt.title("Visual Serial Recognition Distractor Accuracy (Overall)")
    plt.xlabel("Distractor Difference")
    plt.ylabel("Accuracy")
    plt.xticks(distractor_values_sorted)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, output_filename))
    plt.close()


def _plot_vsrec_list_length_by_serial_position_collapsed_over_distractor(
    task_data,
    save_folder,
    output_filename,
):
    """
    VSRec: serial position curves by list length, collapsed over distractor difference.
    Aggregation: sum(correct)/sum(total) for each (list_length, serial_position).
    """
    list_length_tag = f"_{Specs.LIST_LENGTH.value}_"
    serial_position_tag = f"_{Specs.SERIAL_POSITION.value}_"

    # (list_length, serial_position) -> [correct_sum, total_sum]
    aggregated_counts: Dict[Tuple[int, int], List[float]] = {}
    list_lengths_present = set()

    for key, (num_correct, num_total) in task_data.items():
        if list_length_tag not in key or serial_position_tag not in key:
            continue

        list_length = _extract_int_from_key(key, list_length_tag)
        serial_position = _extract_int_from_key(key, serial_position_tag)
        if list_length is None or serial_position is None:
            continue

        aggregated_counts.setdefault((list_length, serial_position), [0.0, 0.0])
        aggregated_counts[(list_length, serial_position)][0] += num_correct
        aggregated_counts[(list_length, serial_position)][1] += num_total
        list_lengths_present.add(list_length)

    if not aggregated_counts:
        return

    plt.figure(figsize=(8, 6))

    for list_length in sorted(list_lengths_present):
        position_points = []
        for (L, pos), (correct_sum, total_sum) in aggregated_counts.items():
            if L != list_length or total_sum <= 0:
                continue
            position_points.append((pos, correct_sum / total_sum))

        if not position_points:
            continue

        position_points.sort()
        x_positions, y_accuracies = zip(*position_points)
        plt.plot(x_positions, y_accuracies, marker="o", label=f"List Length: {list_length}")

    plt.title("VSRec Task Accuracy (List Length x Serial Position)")
    plt.xlabel("Serial Position")
    plt.ylabel("Accuracy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, output_filename))
    plt.close()


def _plot_free_recall_strategy_bars(
    task: Tasks,
    task_data,
    save_folder,
    output_filename,
):
    """
    Categorical bar plot for Spatial Free Recall strategy conditions.
    """
    task_prefix = f"{task.value}_"
    allowed_conditions = ["Forward Order", "No Order", "Recall Error", "First Item Match"]

    category_labels = []
    category_accuracies = []

    for key, (num_correct, num_total) in task_data.items():
        label = key.replace(task_prefix, "")
        if label not in allowed_conditions:
            continue
        if num_total <= 0:
            continue
        category_labels.append(label.title())
        category_accuracies.append(num_correct / num_total)

    if not category_labels:
        return

    plt.figure(figsize=(8, 6))
    bars = plt.bar(category_labels, category_accuracies)
    plt.title(f"{task.value}: Order of Recall Analysis")
    plt.ylabel("Mean Accuracy")
    plt.ylim(0, 1.05)

    for bar in bars:
        value = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}", ha="center", va="bottom")

    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, output_filename))
    plt.close()


def _plot_scalar_accuracy(
    task_data,
    spec: Specs,
    save_folder,
    title,
    filename,
    y_label = "Accuracy",
):
    points = _collect_scalar_points(task_data, spec)
    if not points:
        return

    points = _dedupe_and_sort_points(points)
    _plot_line(
        points=points,
        title=title,
        x_label=spec.value,
        y_label=y_label,
        save_path=os.path.join(save_folder, filename),
    )


def _collect_scalar_points(task_data, spec: Specs):

    spec_tag = f"_{spec.value}_"
    points: List[Tuple[int, float]] = []

    for key, (num_correct, num_total) in task_data.items():
        if spec_tag not in key:
            continue
        if num_total <= 0:
            continue

        spec_value = _extract_int_from_key(key, spec_tag)
        if spec_value is None:
            continue

        accuracy = num_correct / num_total
        points.append((spec_value, accuracy))

    return points


def _sorted_unique_ints_from_keys(keys, tag):
    values = set()
    for key in keys:
        value = _extract_int_from_key(key, tag)
        if value is not None:
            values.add(value)
    return sorted(values)


def _dedupe_and_sort_points(points: List[Tuple[int, float]]):

    tmp = {}
    for x, y in points:
        tmp[int(x)] = float(y)
    return sorted(tmp.items(), key=lambda p: p[0])


def _plot_line(
    points: List[Tuple[int, float]],
    title,
    x_label,
    y_label,
    save_path,
):
    if not points:
        return

    x_values, y_values = zip(*points)
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, marker="o")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(x_values)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _extract_int_from_key(key, tag):
    # key = "..._list_length_6_serial_position_2..."
    # tag = "_list_length_"
    # -> 6

    if tag not in key:
        return None
    try:
        return int(key.split(tag)[1].split("_")[0])
    
    except Exception:
        return None
