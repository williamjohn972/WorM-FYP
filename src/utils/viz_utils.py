import os
from matplotlib import pyplot as plt
from src.tasks import Tasks
from src.config import Config
from src.utils.train_utils import Specs

serial_position = lambda task, task_data, save_path: plot_serial_position(task=task, task_data=task_data, save_path=save_path)
integration = lambda task, task_data, save_path: plot_integration(task=task, task_data=task_data, save_path=save_path)
interaction = lambda task, task_data, save_path: plot_interaction(task=task, task_data=task_data, save_path=save_path)
coordinate_complexity = lambda task, task_data, save_path: plot_coordinate_complexity(task=task, task_data=task_data, save_path=save_path)
categorical_bar = lambda task, task_data, save_path: plot_categorical_bar(task=task, task_data=task_data, save_path=save_path)

PLOT_TYPE_MAP = {
    Tasks.SPATIAL_COORDINATION: [coordinate_complexity],
    Tasks.SPATIAL_FREE_RECALL: [serial_position, categorical_bar],
    Tasks.SPATIAL_INTEGRATION: [integration],
    Tasks.SPATIAL_MEMORY_UPDATING: [serial_position],
    Tasks.SPATIAL_TASK_SWITCHING: [], 

    Tasks.VISUAL_ITEM_RECOGNITION: [interaction, serial_position],
    Tasks.VISUAL_SERIAL_RECALL: [serial_position],
    Tasks.VISUAL_SERIAL_RECOGNITION: [serial_position],

    Tasks.CHANGE_DETECTION_COLOR: [interaction],
    Tasks.CHANGE_DETECTION_ORIENTATION: [interaction],
    Tasks.CHANGE_DETECTION_SIZE: [interaction],
    Tasks.CHANGE_DETECTION_GAP: [interaction],
    Tasks.CHANGE_DETECTION_CONJ: [interaction],
}

def viz_results(epoch, detailed_acc, config: Config, task_list):

    save_path = os.path.join(config.path_config.output_folder, f"epoch_{str(epoch).zfill(3)}")
    os.makedirs(save_path, exist_ok=True)

    for task in task_list:
        plot_fns = PLOT_TYPE_MAP.get(task)

        if isinstance(plot_fns, list):
            task_data = {k: v for k, v in detailed_acc.items() if task.value in k}
            
            for fn in plot_fns:
                fn(task, task_data, save_path)

      
def plot_serial_position(task: Tasks, task_data: dict, save_path):
    
    # Determine which spec to use depending on the task
    if task == Tasks.VISUAL_ITEM_RECOGNITION:
        spec = Specs.RETENTION_INTERVAL

    elif task == Tasks.VISUAL_SERIAL_RECOGNITION:
        spec = Specs.LIST_LENGTH

    elif task == Tasks.SPATIAL_MEMORY_UPDATING:
        spec = Specs.SET_SIZE

    else:
        spec = Specs.LIST_LENGTH

    # Determine if there is a secondary spec we want to track
    if task == Tasks.VISUAL_SERIAL_RECOGNITION:
        secondary_spec = Specs.DISTRACTOR_DIFF
    else:
        secondary_spec = None

    # Eg: "VSR_Task_List_Length_6_Serial_Position_1"
    spec_tag = f"_{spec.value}_"
    primary_spec_values = sorted(list(set(int(
        k.split(spec_tag)[1].split("_")[0]) 
        for k in task_data.keys() if spec_tag in k)))

    plt.figure(figsize=(8,6))
    
    for p_val in primary_spec_values:
        # Filter data for this primary spec value (e.g., List Length 6)
        primary_filtered = {k: v for k, v in task_data.items() if f"{spec_tag}{p_val}" in k}

        # Determine sub-groups (Secondary Spec values or just [None] if no secondary)
        if secondary_spec:
            sec_tag = f"_{secondary_spec.value}_"
            sec_values = sorted(list(set(int(k.split(sec_tag)[1].split("_")[0]) 
                                         for k in primary_filtered.keys() if sec_tag in k)))
        else:
            sec_values = [None] # This allows us to use one loop regardless of task

        # 3. Single Unified Plotting Loop
        for s_val in sec_values:
            points = []
            
            # If secondary exists, filter further; otherwise use primary data
            if s_val is not None and secondary_spec is not None:
                final_data = {k: v for k, v in primary_filtered.items() if f"_{secondary_spec.value}_{s_val}" in k}
                label = f"{spec.value} {p_val}, {secondary_spec.value} {s_val}"
            else:
                final_data = primary_filtered
                label = f"{spec.value}: {p_val}"

            # Extract Position Data
            pos_tag = f"_{Specs.SERIAL_POSITION.value}_"
            for k, v in final_data.items():
                if pos_tag in k:
                    pos_val = int(k.split(pos_tag)[1].split("_")[0])
                    points.append((pos_val, v[0]/v[1]))

            if points:
                points.sort()
                x, y = zip(*points)
                plt.plot(x, y, marker="o", label=label)

    plt.title(f"{task.value.title()} Serial Position Analysis")
    plt.xlabel("Serial Position")
    plt.ylabel("Accuracy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{task.value.title()}_{spec.value.title()}_Serial_Position.png"))
    plt.close()


def plot_interaction(task:Tasks, task_data:dict, save_path):
    
    ri_tag = f"_{Specs.RETENTION_INTERVAL.value}_"
    ris = sorted(list(set(int(
        k.split(ri_tag)[1].split("_")[0])
        for k in task_data.keys() if ri_tag in k)))
    
    plot_save_filename = f"{task.value.title()}_{Specs.RETENTION_INTERVAL.value.title()}_Accuracy.png"
    
    plt.figure(figsize=(8,6))
    
    for ri in ris:
        points = []

        for k, v in task_data.items():
            if f"{ri_tag}{ri}" in k and f"_{Specs.SET_SIZE.value}_" in k:
                set_size = int(k.split(f"_{Specs.SET_SIZE.value}_")[1].split("_")[0])
                correct, total = v[0], v[1]
                points.append((set_size, correct/total))

                plot_save_filename = f"{task.value.title()}_{Specs.RETENTION_INTERVAL.value.title()}_{Specs.SET_SIZE.value.title()}_Accuracy.png"

        if points:
            points.sort()
            x, y = zip(*points)
            plt.plot(x, y, marker="s", label=f"Retention Interval: {ri}s")

    plt.title(f"{task.value.title()} Set Size Vs Retention Interval")
    plt.xlabel("Set Size")
    plt.ylabel("Accuracy")
    plt.legend(title = "Retention Interval", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, plot_save_filename))
    plt.close()


def plot_integration(task:Tasks, task_data:dict, save_path):
    
    spec_tag = f"_{Specs.PART_SIZE.value}_"


    points = []
    for k, v in task_data.items():
        if spec_tag in k:
            part_size = int(k.split(spec_tag)[1].split("_")[0])

            num_integer = (12 // part_size) - 1

            correct, total = v[0], v[1]
            points.append((num_integer, correct/total))

    if points:
        points.sort()
        x, y = zip(*points)
        plt.plot(x, y, marker="o", linestyle="-", color="tab:green", linewidth=2)

        plt.title(f"{task.value}: Complexity Vs Accuracy")
        plt.xlabel("Number of Integrations")
        plt.ylabel("Accuracy")
        plt.xticks(x)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{task.value.title()}_Num_Integration_Accuracy.png"))
        plt.close()

        
def plot_coordinate_complexity(task: Tasks, task_data, save_path):

    metrics_to_plot = [Specs.SET_SIZE, Specs.SYMETRY_OFFSET]

    for spec in metrics_to_plot:
        points = []
        tag = f"_{spec.value}_"
        
        for k, v in task_data.items():
            if tag in k:
                try:
                    val = int(k.split(tag)[1].split('_')[0])
                    
                    correct, total = v[0], v[1]
                    points.append((val, correct/total))
                except (ValueError, IndexError):
                    continue
        
        if not points:
            continue

        points.sort()
        x, y = zip(*points)

        plt.figure(figsize=(8, 8))
        
        color = 'tab:blue' if spec == Specs.SET_SIZE else 'tab:red'
        plt.plot(x, y, marker='o', linestyle='-', color=color, linewidth=2)
        
        plt.title(f"{task.value.title()}: {spec.value} vs. Accuracy")
        plt.xlabel(spec.value)
        plt.ylabel("Accuracy")
        plt.xticks(x)
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        filename = f"{task.value.title()}_{spec.value.title()}_Accuracy.png"
        plt.savefig(os.path.join(save_path, filename))
        plt.close()


def plot_categorical_bar(task: Tasks, task_data: dict, save_path):
    
    task_prefix = f"{task.value}_"

    categories = []
    accuracies = []

    sfr_conditions = ["Forward Order", "No Order", "Recall Error", "First Item Match"]
    
    for key, v in task_data.items():
        label = key.replace(task_prefix, "")

        if label in sfr_conditions:
            correct, total = v[0], v[1]
            categories.append(label.title())
            accuracies.append(correct / total)

    if not categories:
        return

    plt.figure(figsize=(8, 6))
    colors = ['tab:orange', 'tab:green'] 
    
    bars = plt.bar(categories, accuracies, color=colors)
    
    plt.title(f"{task.value}: Order of Recall Analysis")
    plt.ylabel("Mean Accuracy")
    plt.ylim(0, 1.05) 
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')

    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{task.value.title()}_Strategy_Comparison.png"))
    plt.close()