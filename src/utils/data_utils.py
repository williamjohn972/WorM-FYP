from src.data.dataset import *
from torch.utils.data import DataLoader, random_split

from src.tasks import Tasks

CHANGE_DETECTION_COMMON_MAP = dict(
    set_size_options=[2, 4, 6, 8, 10, 12], held_out_set_size_options=[],
    retention_interval_options=[0, 6, 12, 18], held_out_retention_interval_options=[],
)

VISUAL_SERIAL_TASK_COMMON_MAP = dict(
    grid_size = 6,
    list_length_options=[2, 3, 4, 5, 6, 7, 8, 9],
    held_out_list_length_options=[],
)

TASK_DATASET_MAP = {
    Tasks.SPATIAL_COORDINATION: Spatial_Coordination_Dataset,
    Tasks.SPATIAL_FREE_RECALL: Spatial_Free_Recall_Dataset,
    Tasks.SPATIAL_INTEGRATION: Spatial_Integration_Dataset,
    Tasks.SPATIAL_MEMORY_UPDATING: Spatial_Memory_Updating_Dataset,
    Tasks.SPATIAL_TASK_SWITCHING: Spatial_Task_Switching_Dataset, 

    Tasks.VISUAL_ITEM_RECOGNITION: Visual_Item_Recognition_Dataset,
    Tasks.VISUAL_SERIAL_RECALL: Visual_Serial_Task_Dataset,
    Tasks.VISUAL_SERIAL_RECOGNITION: Visual_Serial_Task_Dataset,

    Tasks.CHANGE_DETECTION_COLOR: Change_Detection_Dataset,
    Tasks.CHANGE_DETECTION_ORIENTATION: Change_Detection_Dataset,
    Tasks.CHANGE_DETECTION_SIZE: Change_Detection_Dataset,
    Tasks.CHANGE_DETECTION_GAP: Change_Detection_Dataset,
    Tasks.CHANGE_DETECTION_CONJ: Change_Detection_Dataset,
}

TASK_SPECS_MAP = {
    
    Tasks.SPATIAL_COORDINATION: dict(
        grid_size = 10,                 
        list_length_options=[10, 12, 14, 16, 18], held_out_list_length_options = [],        
        symetry_offset_options=[2,4,6,8], held_out_symetry_offset_options=[]
    ),

    Tasks.SPATIAL_FREE_RECALL: dict(
        grid_size=10, 
        set_size_options=[30], list_length_options=[1,2,3,4,5,6,7,8,10,12,15,18],
        held_out_set_size_options=[], held_out_list_length_options=[],
    ), 

    Tasks.SPATIAL_INTEGRATION: dict(
        grid_size_options=[4], held_out_grid_size_options=[],
        pattern_size_options=[12], held_out_pattern_size_options=[],
        part_size_options=[1,2,3,4,6,12], held_out_part_size_options=[],
        distractor_difference_options=[1,2,3,4], held_out_distractor_difference_options=[],
        max_retries=50,
    ),

    Tasks.SPATIAL_MEMORY_UPDATING: dict(
        grid_size=3,
        set_size_options=[1, 2, 3, 4, 5, 6, 7, 8], held_out_set_size_options=[],
        num_updates_options=[8], presentation_time_options=[1],
    ),

    Tasks.SPATIAL_TASK_SWITCHING: dict(
        trial_length_options=[20],
        held_out_trial_length_options=[],
        variant=STS_Variant.CUED,
    ), 

    Tasks.VISUAL_ITEM_RECOGNITION: dict(
        grid_size=6,
        list_length_options=[4, 6, 8, 10], retention_intervals_options=[0, 2, 4, 5, 6],
        held_out_list_length_options=[], held_out_retention_intervals_options=[],
        distractor_diff_options=[4],
    ),

    Tasks.VISUAL_SERIAL_RECALL: dict(
        variant=VSR_Variant.RECALL,
        distractor_diff_options=[],
        **VISUAL_SERIAL_TASK_COMMON_MAP
    ),

    Tasks.VISUAL_SERIAL_RECOGNITION: dict(
        variant=VSR_Variant.RECOGNITION,
        distractor_diff_options=[2, 4, 6, 8, 10],
        **VISUAL_SERIAL_TASK_COMMON_MAP
    ),

    Tasks.CHANGE_DETECTION_COLOR: dict(
        variant = CD_Variant.COLOR,
        **CHANGE_DETECTION_COMMON_MAP,
    ),

    Tasks.CHANGE_DETECTION_ORIENTATION: dict(
        variant = CD_Variant.ORIENTATION,
        **CHANGE_DETECTION_COMMON_MAP,
    ),
    Tasks.CHANGE_DETECTION_SIZE: dict(
        variant = CD_Variant.SIZE,
        **CHANGE_DETECTION_COMMON_MAP,  
    ),

    Tasks.CHANGE_DETECTION_GAP: dict(
        variant = CD_Variant.GAP,
        **CHANGE_DETECTION_COMMON_MAP,
    ),
    Tasks.CHANGE_DETECTION_CONJ: dict(
        variant = CD_Variant.CONJUNCTION,
        **CHANGE_DETECTION_COMMON_MAP,
    ),
}

# Generate Datasets 
def build_datasets(config):

    datasets = {}

    base_kwargs = dict(
        data_path= config.path_config.data_folder, 
        max_seq_len= config.model_config.max_seq_len, 

        resize_img_size= config.model_config.resize_img_size, 
        img_size = config.model_config.img_size, 

        num_samples =  config.train_config.samples_per_task,

        save = True   
    )

    for task in config.task_config.task_list:

        spec = TASK_SPECS_MAP[task]

        dataset = TASK_DATASET_MAP[task]
        task_kwargs = spec

        train_val_dataset =  dataset(split = "train", **base_kwargs, **task_kwargs)
        
        train_set, val_set = split_train_val_dataset(train_val_dataset)
        test_set =  dataset(split = "test", **base_kwargs, **task_kwargs)

        datasets[task] = {
            "train": train_set,
            "val": val_set,
            "test": test_set,
            }
    

        if config.execution_config.gen_test:
            gen_test_set = dataset(split = "gen_test", **base_kwargs, **task_kwargs)
            datasets[task]["gen_test"] = gen_test_set

    return datasets

# always does a 90/10 split 

def split_train_val_dataset(train_val_dataset): 

    total_length = len(train_val_dataset)
    train_length = int(0.9 * total_length)
    val_length = total_length - train_length

    train_dataset, val_dataset = random_split(train_val_dataset, lengths = [train_length, val_length])

    return train_dataset, val_dataset




# helper function to always turn dataset into train/val loader 
# train_loader: shuffle = True, drop_last = True
# val_loader: shiffle = False, drop_last = False
# a helper that builds test/gen_test loaders 
# test/gen_test loader: shuffle: False, drop_last = False

def build_dataloaders(config):

    datasets = build_datasets(config)

    dataloaders = {}

    for task in config.task_config.task_list:

        dataset = datasets[task]

        dataloaders[task] = {
            "train": DataLoader(dataset["train"], shuffle=True, drop_last=True, batch_size=config.train_config.batch_size, num_workers=config.optimization_config.num_workers, persistent_workers=True, pin_memory=True),
            "val": DataLoader(dataset["val"], shuffle=False, drop_last=False,batch_size=config.train_config.batch_size, num_workers=config.optimization_config.num_workers, persistent_workers=True, pin_memory=True),
            "test": DataLoader(dataset["test"], shuffle=False, drop_last=False,batch_size=config.train_config.batch_size, num_workers=config.optimization_config.num_workers, persistent_workers=True, pin_memory=True),
        }

        if config.execution_config.gen_test:
            dataloaders[task]["gen_test"] = DataLoader(dataset["gen_test"], shuffle=False, drop_last=False,batch_size=config.train_config.batch_size, num_workers=config.optimization_config.num_workers, persistent_workers=True, pin_memory=True)

    return dataloaders
