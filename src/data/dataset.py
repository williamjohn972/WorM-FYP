import os 
import json 
import torch
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

from src.tasks import Tasks


from src.data.generator import Generator
from src.data.spatial_free_recall import Spatial_Free_Recall_Generator
from src.data.spatial_integration import Spatial_Integration_Generator
from src.data.spatial_coordination import Spatial_Coordination_Generator
from src.data.spatial_task_switching import Spatial_Task_Switching_Generator, Variant as STS_Variant
from src.data.spatial_memory_updating import Spatial_Memory_Updating_Generator
from src.data.visual_item_recognition import Visual_Item_Recognition_Generator
from src.data.visual_serial_recall_recognition import Visual_Serial_Task_Generator, Variant as VSR_Variant
from src.data.change_detection import Change_Detection_Generator, Config as CD_Config, Variant as CD_Variant
# from src.data.complex_span import Complex_Span_Generator



class Task_Dataset(Dataset):

    def __init__(self,
                 data_path, 
                 task_name, 
                 split, 
                 resize_img_size, 
                 max_seq_len,
                 data_generator = None,
                 data_generator_kwargs = None):
        

        assert split in ["train", "val", "test", "gen_test"]

        self.data_path = data_path
        self.task_name = task_name
        self.split = split

        self.resize_img_size = resize_img_size
        self.max_seq_len = max_seq_len

        self.task_data_path = os.path.join(data_path, task_name)

        # Generate Data if it does not Exist
        if not os.path.exists(self.task_data_path):
            if data_generator is None:
                raise RuntimeError(f"Data for task {task_name} is not found")
            
            print(f"Generating data for {task_name} ...")
            if data_generator_kwargs is None:
                data_generator_kwargs = {}

            data_generator(**data_generator_kwargs)

        else:
            print("Data already exists. Skipping Data Generation")
            print(f"Getting data for {task_name}")
            print(f"---------------------\n")


        # Load the Trial JSON
        json_path = os.path.join(self.task_data_path, "trials_data.json")
        with open(json_path, "r") as f:
            self.data_file = json.load(f)

        self.data_map = self.data_file[self.split]

        # Image Root for this split
        self.img_path = os.path.join(self.task_data_path, self.split)

        # Create a Consisten Transformation for the images 
        self.transform = transforms.Compose([
            transforms.Resize((resize_img_size, resize_img_size)),
            transforms.ToTensor(),
        ])


    def load_image(self, fname):
        """
        Load RGB Image and apply the transform
        """

        img_path = os.path.join(self.img_path, fname)
        img = Image.open(img_path).convert('RGB')
        return self.transform(img)

    def pad_img_seq(self, img_seq):
        """
        Pad image sequence to max_sequence len using zeros_like
        """

        seq_len = len(img_seq)
        while len(img_seq) < self.max_seq_len:
            img_seq.append(torch.zeros_like(img_seq[0]))
        
        return torch.stack(img_seq), seq_len

    def pad_gt(self, gt, pad_token):
        """
        Pad gt list to max_seq_len using task specific pad_token
        """

        while len(gt) < self.max_seq_len:
            gt.append(pad_token)

        return torch.tensor(gt, dtype=torch.long)

    def __len__(self):
        return len(self.data_map)
    

class Spatial_Integration_Dataset(Task_Dataset):
    """
    returns --> img_seq, gt, seq_len, part_size
    """

    def __init__(self,
                 data_path, 
                 split, 
                 resize_img_size=224, 
                 max_seq_len = 20,
                 
                 num_samples=20,
                 img_size=224,
                 grid_size_options=[4], held_out_grid_size_options=[4],
                 pattern_size_options=[12], held_out_pattern_size_options=[12],
                 distractor_difference_options=[1], held_out_distractor_difference_options=[1],
                 part_size_options=[3,4,6], held_out_part_size_options=[3,4,6],
                 max_retries=13,
                 save=True):
        
        data_generator_kwargs = dict(
            num_samples = num_samples,
            img_size = img_size,
            grid_size_options = grid_size_options, held_out_grid_size_options = held_out_grid_size_options,
            pattern_size_options = pattern_size_options, held_out_pattern_size_options = held_out_pattern_size_options,
            distractor_difference_options = distractor_difference_options, held_out_distractor_difference_options = held_out_distractor_difference_options,
            part_size_options = part_size_options, held_out_part_size_options = held_out_part_size_options,
            max_retries = max_retries,
            data_dir = data_path,save = save
        )

        super().__init__(data_path = data_path, 
                         task_name = Tasks.SPATIAL_INTEGRATION.name.lower(), 
                         split = split, 
                         resize_img_size = resize_img_size, 
                         max_seq_len = max_seq_len, 
                         data_generator = Spatial_Integration_Generator, 
                         data_generator_kwargs = data_generator_kwargs)
        
        self.gt_pad_token = 2
        
        
    
    def __getitem__(self, index):

        trial = self.data_map[index]

        img_seq = []
        gt = []
        
        # Memory Length images with gt pad token
        memory_fnames = trial["memory_stim_fnames"]
        for fname in memory_fnames:
            img_seq.append(self.load_image(fname))

            gt.append(self.gt_pad_token)

        # 1 Probe image
        probe_fname = trial["probe_stim_fnames"][0]
        img_seq.append(self.load_image(probe_fname))
        gt.append(trial["gt"])

        # Seq Len before padding
        img_seq, seq_len = self.pad_img_seq(img_seq)
        
        # pad gt with token 2 to max_seq_len
        gt = self.pad_gt(gt, pad_token= self.gt_pad_token)

        # original returns part_size as a tensor scalar
        part_size = torch.tensor(trial["part_size"])


        return img_seq, gt, seq_len, part_size


class Spatial_Task_Switching_Dataset(Task_Dataset):
    """
    return --> img_seq, gt, seq_len
    """

    def __init__(self,
                data_path, split, resize_img_size, max_seq_len,
                
                img_size,
                trial_length_options = [],
                held_out_trial_length_options = [],
                num_samples = 0,
                variant = None,
                save=True):
        
        data_generator_kwargs = dict(
            num_samples = num_samples,
            img_size = img_size,
            trial_length_options = trial_length_options,
            held_out_trial_length_options = held_out_trial_length_options,
            variant = variant,
            data_dir = data_path, save = save
        )

        super().__init__(data_path = data_path, 
                         task_name = f"{Tasks.SPATIAL_TASK_SWITCHING.name.lower()} - {variant.value.lower()}", 
                         split = split, 
                         resize_img_size = resize_img_size, 
                         max_seq_len = max_seq_len, 
                         data_generator = Spatial_Task_Switching_Generator, 
                         data_generator_kwargs = data_generator_kwargs)
        
        self.gt_pad_token = 2

        self.TASK_TO_INT = {
            "cue_up_down": 2,
            "cue_left_right": 2,
            "up_down": 0,
            "left_right": 1
        }
        
        
    def __getitem__(self, index):

        trial = self.data_map[index]

        img_seq = []
        gt = []
        task_order = []

        stim_fnames = trial["stim_fnames"]
        gt_list = trial["gt_order"]
        task_list = trial["task_order"]

        for fname, label, task in zip(stim_fnames, gt_list, task_list):
            img_seq.append(self.load_image(fname))
            gt.append(int(label))
            task_order.append(self.TASK_TO_INT[task])

        # seq_len before padding
        img_seq, seq_len = self.pad_img_seq(img_seq)

        # pad gtto max_seq_len 
        gt = self.pad_gt(gt, pad_token=self.gt_pad_token)

        # pad task order as well
        task_order = self.pad_gt(task_order, pad_token=self.gt_pad_token)

        return img_seq, gt, seq_len, task_order


class Spatial_Coordination_Dataset(Task_Dataset):

    """
    returns -> img_seq, gt, seq_len, torch.tensor(symetry_offset)  
    if symetry_offset is None return -> img_seq, gt,
    """
    
    def __init__(self,data_path,split, resize_img_size, max_seq_len,
                
                grid_size=5, img_size=224,                 
                list_length_options=[], held_out_list_length_options=[],        
                symetry_offset_options=[],    
                held_out_symetry_offset_options=[],
                num_samples=0,
                save=True,
            ):
        
        data_generator_kwargs = dict(
            num_samples = num_samples,
            img_size = img_size,
            grid_size = grid_size,
            list_length_options = list_length_options,
            symetry_offset_options = symetry_offset_options,
            held_out_list_length_options = held_out_list_length_options,
            held_out_symetry_offset_options = held_out_symetry_offset_options,
            data_dir = data_path, save = save
        )

        super().__init__(data_path = data_path, 
                         task_name = f"{Tasks.SPATIAL_COORDINATION.name.lower()}", 
                         split = split, 
                         resize_img_size = resize_img_size, 
                         max_seq_len = max_seq_len, 
                         data_generator = Spatial_Coordination_Generator, 
                         data_generator_kwargs = data_generator_kwargs)
        
        self.gt_pad_token = 2
        
        
    def __getitem__(self, index):

        trial = self.data_map[index]

        img_seq = []
        gt = []

        memory_stim_fnames = trial["memory_stim_fnames"]
        probe_stim_fname = trial["probe_stim_fnames"][0]
        label = trial["gt"]
        symetry_offset = trial["symetry_offset"]


        # Memory Phase
        for fname in memory_stim_fnames:
            img_seq.append(self.load_image(fname))
            gt.append(self.gt_pad_token)

        # Probe Phase
        img_seq.append(self.load_image(probe_stim_fname))
        gt.append(int(label))

        # seq_len BEFORE Padding
        img_seq, seq_len = self.pad_img_seq(img_seq)

        # pad gt
        gt = self.pad_gt(gt, self.gt_pad_token)

        if symetry_offset is None:
            return img_seq, gt, seq_len, torch.tensor(-1)
        
        return img_seq, gt, seq_len, torch.tensor(symetry_offset)


class Visual_Item_Recognition_Dataset(Task_Dataset):
    """
    return --> img_seq, gt, seq_len, retention_interval, 
    """

    def __init__(self, data_path, split, resize_img_size, max_seq_len,
                
                grid_size=10, img_size=224,
                list_length_options=[], retention_intervals_options=[],
                held_out_list_length_options=[], held_out_retention_intervals_options=[],
                distractor_diff_options=[],
                num_samples=4,
                save = True,
            ):
        
        data_generator_kwargs = dict(
            num_samples = num_samples,
            img_size = img_size,
            grid_size = grid_size,
            list_length_options = list_length_options,
            held_out_list_length_options = held_out_list_length_options,
            retention_intervals_options = retention_intervals_options,
            held_out_retention_intervals_options = held_out_retention_intervals_options,
            distractor_diff_options = distractor_diff_options,
            data_dir = data_path, save = save
        )

        super().__init__(data_path = data_path, 
                         task_name = f"{Tasks.VISUAL_ITEM_RECOGNITION.name.lower()}", 
                         split = split, 
                         resize_img_size = resize_img_size, 
                         max_seq_len = max_seq_len, 
                         data_generator = Visual_Item_Recognition_Generator, 
                         data_generator_kwargs = data_generator_kwargs)
        
        self.gt_pad_token = 2
        
        
    def __getitem__(self, index):

        trial = self.data_map[index]

        img_seq = []
        gt = []

        memory_stim_fnames = trial["memory_stim_fnames"]
        retention_stim_fnames = trial["retention_stim_fnames"]
        probe_stim_fname = trial["probe_stim_fnames"][0]
        
        label = trial["recall_gt"]
        gt_index = int(trial["recall_gt_index"])
        retention_interval = int(trial["retention_interval"])


        # Memory Phase
        for fname in memory_stim_fnames:
            img_seq.append(self.load_image(fname))
            gt.append(self.gt_pad_token)

        # Retention Phase
        for fname in retention_stim_fnames:
            img_seq.append(self.load_image(fname))
            gt.append(self.gt_pad_token)

        # Probe Phase
        img_seq.append(self.load_image(probe_stim_fname))
        gt.append(int(label))

        # seq_len BEFORE Padding
        img_seq, seq_len = self.pad_img_seq(img_seq)

        # pad gt
        gt = self.pad_gt(gt, self.gt_pad_token)

        if gt_index is None:
            gt_index = torch.tensor(-1)

        else:
            gt_index = torch.tensor(gt_index)
        
        return img_seq, gt, seq_len, torch.tensor(retention_interval), gt_index


class Change_Detection_Dataset(Task_Dataset):

    """
    return -->  img_seq, gt, seq_len, retention_interval, set_size

    if Conjuction return --> return img_seq, gt, seq_len, retention_interval, set_size, conj_label
    """

    def __init__(self,
                data_path, split, resize_img_size, max_seq_len,
                
                img_size=224, variant=None,
                num_samples=0,
                set_size_options=[], held_out_set_size_options=[],
                retention_interval_options=[], held_out_retention_interval_options=[],
                save = True,
            ):
        
        
        data_generator_kwargs = dict(
            num_samples = num_samples,
            img_size = img_size,
            variant = variant,
            set_size_options = set_size_options,
            held_out_set_size_options = held_out_set_size_options,
            retention_interval_options = retention_interval_options,
            held_out_retention_interval_options = held_out_retention_interval_options,
            data_dir = data_path, save = save,
        )

        super().__init__(data_path = data_path, 
                         task_name = f"{Tasks.CHANGE_DETECTION.name.lower()} - {variant.value.lower()}", 
                         split = split, 
                         resize_img_size = resize_img_size, 
                         max_seq_len = max_seq_len, 
                         data_generator = Change_Detection_Generator, 
                         data_generator_kwargs = data_generator_kwargs)
        
        self.gt_pad_token = 2
        self.conjunction_none_place_holder = 4

        self.conjunction = True if variant == CD_Variant.CONJUNCTION else False
        
        
    def __getitem__(self, index):

        trial = self.data_map[index]

        img_seq = []
        gt = []

        memory_stim_fname = trial["memory_stim_fnames"][0]
        retention_stim_fnames = trial["retention_stim_fnames"]
        probe_stim_fname = trial["probe_stim_fnames"][0]
        
        label = trial["recall_gt"]
        
        retention_interval = torch.tensor(int(trial["retention_interval"]))

        set_size = torch.tensor(int(trial["set_size"]))

        # Memory Phase
        img_seq.append(self.load_image(memory_stim_fname))
        gt.append(self.gt_pad_token)

        # Retention Phase
        for fname in retention_stim_fnames:
            img_seq.append(self.load_image(fname))
            gt.append(self.gt_pad_token)

        # Probe Phase
        img_seq.append(self.load_image(probe_stim_fname))
        gt.append(int(label))

        # seq_len BEFORE Padding
        img_seq, seq_len = self.pad_img_seq(img_seq)

        # pad gt
        gt = self.pad_gt(gt, self.gt_pad_token)

        if self.conjunction:
            conj_label_str = trial.get("conjunction_gt", None)

            if conj_label_str is None:
                conj_label = torch.tensor(self.conjunction_none_place_holder)
            else:
                conj_label = torch.tensor(CD_Config.CONJUNCTION_GT_MAP[conj_label_str])

            return img_seq, gt, seq_len,  retention_interval, set_size, conj_label

        return img_seq, gt, seq_len, retention_interval, set_size, torch.tensor(-1)


class Visual_Serial_Task_Dataset(Task_Dataset):
    """
    Visual Serial Recall:
            return --> img_seq, gt, seq_len, torch.tensor(list_length)

    Visual Serial Recogition:
            return --> img_seq, gt, seq_len, torch.tensor(list_length), torch.tensor(distractor_diff)
    """

    def __init__(self,data_path, split, resize_img_size, max_seq_len,
                
                grid_size=10,
                img_size=224,
                variant=None,
                list_length_options=[],
                held_out_list_length_options=[],
                distractor_diff_options=[],
                num_samples=[],
                save=True,
            ):
        
        
        data_generator_kwargs = dict(
            num_samples = num_samples,
            img_size = img_size,
            grid_size = grid_size,
            list_length_options = list_length_options,
            held_out_list_length_options = held_out_list_length_options,
            distractor_diff_options = distractor_diff_options,
            variant = variant,
            data_dir = data_path, save = save,
        )

        self.variant = variant
        self.task_name = Tasks.VISUAL_SERIAL_RECALL if self.variant == VSR_Variant.RECALL else Tasks.VISUAL_SERIAL_RECOGNITION

        super().__init__(data_path = data_path, 
                         task_name = f"{self.task_name.name.lower()}", 
                         split = split, 
                         resize_img_size = resize_img_size, 
                         max_seq_len = max_seq_len, 
                         data_generator = Visual_Serial_Task_Generator, 
                         data_generator_kwargs = data_generator_kwargs)
        

        self.gt_pad_token = None
        if self.variant == VSR_Variant.RECALL:
            self.gt_pad_token = 6

        elif self.variant == VSR_Variant.RECOGNITION:
            self.gt_pad_token = 2

        assert self.variant in list(VSR_Variant)

        
        
    def __getitem__(self, index):

        trial = self.data_map[index]

        img_seq = []
        gt = []

        memory_stim_fnames = trial["memory_stim_fnames"]
        probe_stim_fnames = trial["probe_stim_fnames"]
        list_length = trial["list_length"]
        
        # Memory Phase
        for fname in memory_stim_fnames:
            img_seq.append(self.load_image(fname))
            gt.append(self.gt_pad_token)

        # Probe Phase
        if self.variant == VSR_Variant.RECALL:
            recall_gt = trial["recall_gt"]
            probe_stim_fname = probe_stim_fnames[0]


            for _ in range(list_length):
                img_seq.append(self.load_image(probe_stim_fname))
                
            # append the recall targets 
            gt.extend([int(x) for x in recall_gt])

            # seq_len BEFORE Padding
            img_seq, seq_len = self.pad_img_seq(img_seq)

            # pad gt
            gt = self.pad_gt(gt, self.gt_pad_token)

            return img_seq, gt, seq_len, torch.tensor(list_length)


        elif self.variant == VSR_Variant.RECOGNITION:
            recog_gt = trial["recog_gt"]
            distractor_diff = int(trial.get("distractor_diff", -1))

            for fname in probe_stim_fnames:
                img_seq.append(self.load_image(fname))

            gt.extend(int(x) for x in recog_gt)

            # seq_len BEFORE Padding
            img_seq, seq_len = self.pad_img_seq(img_seq)

            # pad gt
            gt = self.pad_gt(gt, self.gt_pad_token)


            return img_seq, gt, seq_len, torch.tensor(list_length), torch.tensor(distractor_diff)


class Spatial_Memory_Updating_Dataset(Task_Dataset):
    """
    return --> img_seq, gt, seq_len, set_size
    """

    def __init__(self, data_path, split, resize_img_size, max_seq_len,
                
                grid_size=10, img_size=224,
                set_size_options=[], held_out_set_size_options=[],
                num_updates_options=[], presentation_time_options=[],
                num_samples=0, 
                save=True
            ):
        
        data_generator_kwargs = dict(
            num_samples = num_samples,
            img_size = img_size,
            grid_size = grid_size,
            set_size_options = set_size_options,
            held_out_set_size_options = held_out_set_size_options,
            num_updates_options = num_updates_options,
            presentation_time_options = presentation_time_options,
            data_dir = data_path, save = save
        )

        super().__init__(data_path = data_path, 
                         task_name = f"{Tasks.SPATIAL_MEMORY_UPDATING.name.lower()}", 
                         split = split, 
                         resize_img_size = resize_img_size, 
                         max_seq_len = max_seq_len, 
                         data_generator = Spatial_Memory_Updating_Generator, 
                         data_generator_kwargs = data_generator_kwargs)
        
        self.gt_pad_token = 9
        
        
    def __getitem__(self, index):

        trial = self.data_map[index]

        img_seq = []
        gt = []

        memory_stim_fname = trial["memory_stim_fnames"][0]
        update_stim_fnames = trial["update_stim_fnames"]
        probe_stim_fnames = trial["probe_stim_fnames"]
        probe_gt_order = trial["probe_gt_order"]

        num_updates = trial["num_updates"]
        presentation_time = trial["presentation_time"]
        set_size = trial["set_size"]

        # Memory Phase
        img_seq.append(self.load_image(memory_stim_fname))
        gt.append(self.gt_pad_token)

        # Update Phase
        for i in range(num_updates):

            cur_update_stim_fname = update_stim_fnames[i]
            for _ in range(presentation_time):
                img_seq.append(self.load_image(cur_update_stim_fname))
                gt.append(self.gt_pad_token)

        # Probe Phase
        for i in range(set_size):
            img_seq.append(self.load_image(probe_stim_fnames[i]))
        
        gt.extend(int (x) for x in probe_gt_order)

        # seq_len BEFORE Padding
        img_seq, seq_len = self.pad_img_seq(img_seq)

        # pad gt
        gt = self.pad_gt(gt, self.gt_pad_token)

        
        return img_seq, gt, seq_len, torch.tensor(set_size)


class Spatial_Free_Recall_Dataset(Task_Dataset):
    """
    return --> img_seq, gt_one_hot, seq_len, recall_gt_original
    """
    def __init__(self, data_path, split, resize_img_size, max_seq_len,
                
                grid_size=10, 
                set_size_options=[], list_length_options=[],
                img_size=224,
                held_out_set_size_options=[], held_out_list_length_options=[],
                num_samples=0,
                save=True
            ):
        
        data_generator_kwargs = dict(
            num_samples = num_samples,
            img_size = img_size,
            grid_size = grid_size,
            set_size_options = set_size_options,
            held_out_set_size_options = held_out_set_size_options,
            list_length_options = list_length_options,
            held_out_list_length_options = held_out_list_length_options,
            data_dir = data_path, save = save
        )

        super().__init__(data_path = data_path, 
                         task_name = f"{Tasks.SPATIAL_FREE_RECALL.name.lower()}", 
                         split = split, 
                         resize_img_size = resize_img_size, 
                         max_seq_len = max_seq_len, 
                         data_generator = Spatial_Free_Recall_Generator, 
                         data_generator_kwargs = data_generator_kwargs)
        
        self.gt_pad_token = -1
        
        
    def __getitem__(self, index):

        trial = self.data_map[index]

        img_seq = []

        memory_stim_fnames = trial["memory_stim_fnames"]
        probe_stim_fname = trial["probe_stim_fname"]
        recall_gt = trial["recall_gt"]

        # Memory Phase
        for fname in memory_stim_fnames:
            img_seq.append(self.load_image(fname))

        # Probe Phase
        img_seq.append(self.load_image(probe_stim_fname))
        

        # seq_len BEFORE Padding
        img_seq, seq_len = self.pad_img_seq(img_seq)

        # used for analysis
        recall_gt_original = list(recall_gt)

        recall_gt_original = self.pad_gt(recall_gt_original, self.gt_pad_token)
        recall_gt_original = torch.tensor(recall_gt_original, dtype=torch.long)
        
        recall_gt_tensor = torch.tensor(recall_gt, dtype=torch.long)

        if recall_gt_tensor.numel() == 0:
            # used for training
            gt_one_hot = torch.zeros(100, dtype=torch.long)
        else:
            gt_one_hot = F.one_hot(recall_gt_tensor, num_classes=100).sum(dim=0).to(torch.long)

        return img_seq, gt_one_hot, seq_len, recall_gt_original
        
