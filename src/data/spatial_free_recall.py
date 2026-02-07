import os
import random
import json
from tqdm.auto import tqdm
from PIL import Image, ImageDraw
from typing import List, Dict

from src.data.generator import Generator, Colors
from src.tasks import Tasks


class Config:

    ACTIVE_CELL_COLOR =  Colors.RED.value
    INACTIVE_CELL_COLOR= Colors.GREEN.value
    CELL_BACKGROUND_COLOR= Colors.GRAY.value

    CELL_OUTLINE_COLOR= Colors.GRAY.value
    CELL_OUTLINE_WIDTH= 2

    CELL_PADDING = 3

    
class Spatial_Free_Recall_Generator(Generator):

    def __init__(self, grid_size:int, list_length_options:List[int], set_size_options:List[int],
                 data_dir:str,
                 img_size:int,
                 held_out_set_size_options: List[int], held_out_list_length_options:List[int],
                 num_samples:int,
                 save:bool = True, generate_trials=True):
        
    
        self.grid_size = grid_size
        self.list_length_options = list_length_options
        self.set_size_options = set_size_options

        self.held_out_set_size_options = held_out_set_size_options
        self.held_out_list_length_options = held_out_list_length_options
        
        self.img_size = (img_size,img_size)

        self.data_dir = data_dir
        self.save = save

        self.config = Config()

        self.train_num_samples = num_samples 
        self.test_num_samples = (int) (num_samples * 0.1)
        self.gen_test_num_samples = (int) (num_samples * 0.1)

        # Initialise the paths 
        self.train_dir, self.test_dir, self.gen_test_dir = self._generate_trial_directories(data_dir,'Spatial Free Recall')
        
        # Assertions
        assert self.train_num_samples % (len(set_size_options)*len(list_length_options)) == 0
        assert self.test_num_samples % (len(set_size_options)*len(list_length_options)) == 0
        if len(self.held_out_set_size_options) > 0 and len(self.held_out_list_length_options) > 0:
            assert self.gen_test_num_samples % (len(self.held_out_set_size_options)*len(self.held_out_list_length_options)) == 0

        assert min(set_size_options) >= max(list_length_options)
        if len(self.held_out_set_size_options) > 0 and len(self.held_out_list_length_options) > 0:
            assert min(self.held_out_set_size_options) >= max(self.held_out_list_length_options)


        if self.save:
            self.train_dir, self.test_dir, self.gen_test_dir = self._init_dirs(self.data_dir, Tasks.SPATIAL_FREE_RECALL.name.lower())

        if generate_trials:
            self.generate_trials()

    def generate_trials(self):

        self._log("Generating trials")        

        trials_dict = {'train': [], 'test': [], 'gen_test': []}

        for trial_type in trials_dict.keys():
            # print(f"Generating {trial_type} trials ...")

            self._log(f"Split: {trial_type}")

            # What will the set_size and list_length options be ?
            if trial_type in ["train", "test"]:
                set_size_options = self.set_size_options
                list_length_options = self.list_length_options

            else:
                set_size_options = self.held_out_set_size_options
                list_length_options = self.held_out_list_length_options

            num_samples = getattr(self, f"{trial_type}_num_samples")

            # Compute Number of trials per condition
            num_samples_per_condition = num_samples // (len(set_size_options) * len(list_length_options))
            per_condition_sample_count = {}
            overall_sample_count = 0

            # Create trials for every combination of set size and list length
            for set_size in tqdm(set_size_options, leave=False, desc=f"{trial_type} | set_size"):
                for list_length in list_length_options:
                    
                    condition = (set_size, list_length)
                    per_condition_sample_count[condition] = 0

                    # Generate trials until num_samples quota is met
                    while(per_condition_sample_count[condition] < num_samples_per_condition):
                        
                        # Sample visible cells
                        visible_cells = random.sample(range(self.grid_size**2), set_size) 

                        # Sample recall sequence of active cells
                        recall_gt = random.sample(visible_cells, list_length)

                        # For each recall sequence create the file name 
                        memory_stim_fnames = []
                        for idx, _ in enumerate(recall_gt):
                            memory_stim_fnames.append(f"{trial_type}_{str(overall_sample_count).zfill(6)}_memory{str(idx).zfill(2)}.png")

                        probe_stim_fname = f"{trial_type}_{str(overall_sample_count).zfill(6)}_probe.png"

                        # Store everything in trials[trial_type]
                        trials_dict[trial_type].append({
                            'trial_type': trial_type,
                            'trial_id': f"{trial_type}_{str(overall_sample_count).zfill(6)}",

                            'grid_size': self.grid_size,
                            'set_size': set_size,
                            'list_length': list_length,

                            'visible_cells': visible_cells,
                            'recall_gt': recall_gt,
                            
                            "memory_stim_fnames": memory_stim_fnames,
                            "probe_stim_fname": probe_stim_fname
                        })

                        # Increment Counts
                        overall_sample_count+=1
                        per_condition_sample_count[condition]+=1

        # Create Trials and Dump them to JSON
        if self.save:
            self._save_trial_json(trials_dict)

        self._draw_trials_stim(trials_dict)

        self._log_summary(trials_dict)


        return trials_dict
    
    def _draw_trials_stim(self,trials):
        
        trial_types = ["train", "test", "gen_test"]
        trials_stim_list = {"train":[], "test": [], "gen_test": []}

        for trial_type in trial_types:
            # print(f"Generating {trial_type} Stimuli")

            for trial in trials[trial_type]:

                memory_stim_list = self.draw_memory_stim(
                    grid_size=trial["grid_size"],
                    visible_cells=trial["visible_cells"],
                    recall_gt=trial["recall_gt"]
                )

                probe_stim = self.draw_probe_stim()

                trial_stim = {}
                trial_stim["memory_stim_list"] = memory_stim_list
                trial_stim["probe_stim"] = probe_stim

                trials_stim_list[trial_type].append(trial_stim)

                # Saving Logic
                if self.save:

                    save_file_path = getattr(self, f"{trial_type}_dir")
                    
                    for memory_stim, memory_stim_fname in zip(memory_stim_list, trial["memory_stim_fnames"]):
                        memory_stim.save(os.path.join(save_file_path,memory_stim_fname))

                    probe_stim.save(os.path.join(save_file_path, trial["probe_stim_fname"]))
                      
        return trials_stim_list


    def draw_memory_stim(self, 
                         grid_size:int,
                         visible_cells:List[int], recall_gt:List[int]):
        
        memory_stim_list = []

        # Recall gt has he sequence of active cells
        for cur_active_cell in recall_gt:

            # We need to create a blank background around it
            memory_stim, draw = self._init_stim(self.img_config.PRESENTATION_OUTLINE_COLOR)


            # Add Margin and Define Cell Size
            grid_size_px = self.img_size[0] - 2*self.config.CELL_PADDING
            grid_cell_size_px = grid_size_px // grid_size

            # Add Visible Cells
            for cell in visible_cells:
                cell_x = cell % grid_size
                cell_y = cell // grid_size

                cell_x_px = cell_x * grid_cell_size_px + self.config.CELL_PADDING
                cell_y_px = cell_y * grid_cell_size_px + self.config.CELL_PADDING
            
                # Draw Cells
                if cell == cur_active_cell:
                    cell_color = self.config.ACTIVE_CELL_COLOR

                else:
                    cell_color = self.config.INACTIVE_CELL_COLOR

                draw.rectangle([cell_x_px, cell_y_px,
                                cell_x_px + grid_cell_size_px, cell_y_px + grid_cell_size_px], 
                                fill=cell_color,
                                outline=self.config.CELL_OUTLINE_COLOR,
                                width=self.config.CELL_OUTLINE_WIDTH)
                
            memory_stim_list.append(memory_stim)
        
        return memory_stim_list


    def draw_probe_stim(self):
        stim, draw = self._init_stim(self.img_config.PROBE_OUTLINE_COLOR)
        
        

        return stim

    




