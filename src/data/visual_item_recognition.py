from src.data.generator import Generator, Colors
from typing import List, Dict, Tuple
import random
import json
from PIL import Image,ImageDraw
from enum import Enum
from tqdm.auto import tqdm

from src.tasks import Tasks

class GroundTruth(Enum):
    LEFT = 0
    RIGHT = 1


class Config:

    MEMORY_STIM_PADDING = 9
    PROBE_STIM_PADDING = 3

    MEMORY_CELL_RADIUS = 3
    PROBE_CELL_RADIUS = 1

    PROBE_LINE_WIDTH = 1

    DARK_CELL_COLOR = Colors.RED.value
    WHITE_CELL_COLOR = Colors.WHITE.value

    CELL_OUTLINE_COLOR = Colors.BLACK.value
    PROBE_LINE_COLOR = Colors.BLACK.value
        

class Visual_Item_Recognition_Generator(Generator):

    def __init__(self,
                 grid_size:int, img_size:int,
                 list_length_options:List[int], retention_intervals_options:List[int],
                 held_out_list_length_options:List[int], held_out_retention_intervals_options:List[int],
                 distractor_diff_options: List[int],
                 num_samples,
                 data_dir:str="",
                 save:bool = True, generate_trials = True,
                ):


        self.grid_size = grid_size
        self.img_size = (img_size, img_size)

        self.config = Config()

        self.list_length_options = list_length_options
        self.retention_intervals_options = retention_intervals_options

        self.held_out_list_length_options = held_out_list_length_options
        self.held_out_retention_intervals_options = held_out_retention_intervals_options

        self.distractor_diff_options = distractor_diff_options

        self.data_dir = data_dir
        self.save = save


        self.train_num_samples = num_samples
        self.test_num_samples = (int) (num_samples * 0.1)
        self.gen_test_num_samples = (int) (num_samples * 0.1)

        if self.save:
            self.train_dir, self.test_dir, self.gen_test_dir = self._init_dirs(self.data_dir, Tasks.VISUAL_ITEM_RECOGNITION.name.lower())

        if generate_trials:
            self.generate_trials()


    def generate_trials(self):
        self._log("Generating trials")        

        trials = {"train": [], "test": [], "gen_test": []}

        for trial_type in ["train", "test" , "gen_test"]:
            self._log(f"Split: {trial_type}")

            
            # List Length and Retention Interval Options are dependant on Trial Type 
            if trial_type in ["train", "test"]:
                list_length_options = self.list_length_options
                retention_interval_opions = self.retention_intervals_options

            else:
                list_length_options = self.held_out_list_length_options
                retention_interval_opions = self.held_out_retention_intervals_options
                

            # Generated Trial Counts 
            # Combination is (list_length, retention interval)
            cur_total_trial_samples = 0
            cur_trial_samples_per_combination = {}

            # Split Total Num of Samples Generated between all possible Combinations
            num_possible_combinations = len(list_length_options) * len(retention_interval_opions)
            total_trial_samples = getattr(self, f"{trial_type}_num_samples")
            total_trial_samples_per_combination = total_trial_samples // max(1,(num_possible_combinations))
            
            
            # We need to Loop over every single combination
        
            for list_length in tqdm(list_length_options, leave=False, desc=f"{trial_type} | list_length"):
                for retention_interval in retention_interval_opions:

                    combination = (list_length, retention_interval)
                    cur_trial_samples_per_combination[combination] = 0

                    while(cur_trial_samples_per_combination[combination] < total_trial_samples_per_combination):
                        
                        num_grid_cells = self.grid_size**2

                        # -- Memory Stim --
                        memory_items = []

                        # For each memory stim in the sequence
                        for _ in range(list_length):

                            # Choose the Half the cells to be dark
                            dark_cells = random.sample(range(num_grid_cells), num_grid_cells // 2)
                            memory_items.append(
                                {
                                    "dark_cells": dark_cells,
                                    "white_cells": [cell for cell in range(num_grid_cells) if cell not in dark_cells]
                                })

                        # Choose which of the memory stims you want to test on 
                        recall_gt_index = random.choice(range(list_length))
                        
                        # Should the correct image be on the left or right ?
                        recall_gt = random.choice(list(GroundTruth))

                        # -- Probe Stim -- 
                        # Time to create the distractors
                        dark_cells = memory_items[recall_gt_index]["dark_cells"]
                        white_cells = memory_items[recall_gt_index]["white_cells"]

                        distractor_diff = random.choice(self.distractor_diff_options)

                        # distractor_diff // 2 dark cells become white 
                        # distractor_diff // 2 white cells become dark 
                        new_white_cells = random.sample(dark_cells, distractor_diff//2)
                        new_dark_cells = random.sample(white_cells, distractor_diff//2)

                        # Flip them 
                        distractor_dark_cells = [cell for cell in dark_cells if cell not in new_white_cells]
                        distractor_dark_cells.extend(new_dark_cells)

                        distractor_white_cells = [cell for cell in white_cells if cell not in new_dark_cells]
                        distractor_white_cells.extend(new_white_cells)

                        # -- Generate File Names --
                        id = cur_total_trial_samples
                        memory_stim_fnames = [
                            f"{trial_type}_{str(id).zfill(6)}_memory{str(idx).zfill(2)}.png" for idx in range(list_length)
                        ]

                        retention_stim_fnames = [
                            f"{trial_type}_{str(id).zfill(6)}_retention{str(idx).zfill(2)}.png" for idx in range(retention_interval)
                        ]

                        probe_stim_fnames = [
                            f"{trial_type}_{str(id).zfill(6)}_probe{str(idx).zfill(2)}.png" for idx in range(1)
                        ]


                        # Adding the trial_data to trials
                        trials[trial_type].append({
                            "trial_type": trial_type,
                            "trial_id": id, 
                            
                            "grid_size": self.grid_size,

                            "list_length": list_length,
                            "retention_interval": retention_interval,

                            "distractor_diff": distractor_diff,
                            "memory_items": memory_items,

                            "distractor_item": {
                                "dark_cells": distractor_dark_cells,
                                "white_cells": distractor_white_cells,
                            },
                            
                            "recall_gt_index": recall_gt_index,  
                            "recall_gt": recall_gt.value,
                            "recall_gt_value": recall_gt.name.lower(),

                            "memory_stim_fnames": memory_stim_fnames,
                            "retention_stim_fnames": retention_stim_fnames,
                            "probe_stim_fnames": probe_stim_fnames,
                        })

                        # Update Counters
                        cur_total_trial_samples += 1
                        cur_trial_samples_per_combination[combination] += 1

        if self.save:
            self._save_trial_json(trials)

        self._draw_trial_stims(trials)

        self._log_summary(trials)

        return trials
    

    def _draw_trial_stims(self, trials:Dict[str,List[Dict]]):

        trial_stim_list = {"train": [], "test": [], "gen_test": []}

        # For Each Trial Type 
        for trial_type in self.trial_types:

            trials_list = trials[trial_type]

            # We go over each trial 
            for trial in trials_list:

                # Draw the memory stim 
                memory_stims = self._draw_memory_stims(trial)

                # Draw the retention stim
                retention_stims = self._draw_retention_stims(trial)

                # Draw the probe stim 
                probe_stims = self._draw_probe_stims(trial)

                # Saving Logic 
                if self.save:
                    self._save_stim(trial_type, memory_stims, trial["memory_stim_fnames"])
                    self._save_stim(trial_type, retention_stims, trial["retention_stim_fnames"])
                    self._save_stim(trial_type, probe_stims, trial["probe_stim_fnames"])


    def _draw_memory_stims(self, trial):

        memory_items = trial["memory_items"]
        
        stims = []

        for memory_item in memory_items:

            dark_cells = memory_item["dark_cells"]
            white_cells = memory_item["white_cells"]

            stim, draw = self._init_stim(self.img_config.PRESENTATION_OUTLINE_COLOR)

            # Calculate the grid_size in px
            # We need to subtract the padding 
            grid_size_px = self.img_size[0] - self.config.MEMORY_STIM_PADDING*2
            
            # Calculate the size of each grid_cell in px
            grid_cell_size_px = grid_size_px // self.grid_size

            offset = (self.img_size[0] - grid_size_px) // 2

            # Draw the dark cells 
            for cell in dark_cells:
                self._draw_grid_cell(draw, 
                                     cell,offset, offset,
                                     grid_cell_size_px, self.config.MEMORY_CELL_RADIUS,
                                     self.config.DARK_CELL_COLOR, self.config.CELL_OUTLINE_COLOR)
                
            # Draw the white cells
            for cell in (white_cells):
                self._draw_grid_cell(draw, 
                                     cell, offset, offset,
                                     grid_cell_size_px, self.config.MEMORY_CELL_RADIUS,
                                     self.config.WHITE_CELL_COLOR, self.config.CELL_OUTLINE_COLOR)

            stims.append(stim)

        return stims
    
    def _draw_grid_cell(self, draw, 
                        cell, x_offset, y_offset, 
                        grid_cell_size_px, radius,
                        fill, outline_color):

        # Calc grid coords
        cell_x = cell % self.grid_size  
        cell_y = cell // self.grid_size
                
        # Calc top left px position
        x1 = cell_x * grid_cell_size_px + x_offset
        y1 = cell_y * grid_cell_size_px + y_offset

        x2 = x1 + grid_cell_size_px
        y2 = y1 + grid_cell_size_px

        draw.rounded_rectangle([x1,y1,x2,y2],
                               radius=radius,
                               fill=fill,
                               outline=outline_color)

    def _draw_retention_stims(self, trial):

        retention_interval = trial["retention_interval"]

        stims = []

        for _ in range(retention_interval):
            stim, _ = self._init_stim(self.img_config.RETENTION_OUTLINE_COLOR)
            stims.append(stim)

        return stims

    def _draw_probe_stims(self, trial):

        memory_items = trial["memory_items"]

        recall_gt_index = trial["recall_gt_index"]
        recall_gt = trial["recall_gt"]

        true_memory = memory_items[recall_gt_index]
        distractor = trial["distractor_item"]

        if recall_gt == 0:  # Left 
            left_stim = true_memory
            right_stim = distractor

        else:
            left_stim = distractor
            right_stim = true_memory


        stims = []

        stim, draw = self._init_stim(self.img_config.PROBE_OUTLINE_COLOR)

        # Each grid is Half the size of the Image - Padding
        grid_size_px = self.img_size[0]//2 - self.config.PROBE_STIM_PADDING*2 
        grid_cell_size_px = grid_size_px // self.grid_size


        actual_grid_width = self.grid_size * grid_cell_size_px

        internal_margin = (grid_size_px - actual_grid_width) // 2

        y_offset = (self.img_size[0] - actual_grid_width) // 2
        half_width = self.img_size[0] // 2

        for idx in range(2):
        
            if idx == 0:
                cur_stim = left_stim
                x_offset = self.config.PROBE_STIM_PADDING + internal_margin

            else:
                cur_stim = right_stim
                x_offset = half_width + self.config.PROBE_STIM_PADDING + internal_margin

            
            dark_cells = cur_stim["dark_cells"]
            white_cells = cur_stim["white_cells"]
            
            for cell in dark_cells:
                self._draw_grid_cell(draw,cell,x_offset,y_offset,
                                    grid_cell_size_px, self.config.PROBE_CELL_RADIUS,
                                    self.config.DARK_CELL_COLOR, self.config.CELL_OUTLINE_COLOR)

            for cell in white_cells:
                self._draw_grid_cell(draw,cell,x_offset,y_offset,
                                    grid_cell_size_px, self.config.PROBE_CELL_RADIUS,
                                    self.config.WHITE_CELL_COLOR, self.config.CELL_OUTLINE_COLOR)
        
        draw.line([half_width, 20, half_width, self.img_size[1]-20],
                  fill=self.config.PROBE_LINE_COLOR,
                  width=self.config.PROBE_LINE_WIDTH)
        

        # Draw Right Grid 
        stims.append(stim)

        return stims
        
        






        

