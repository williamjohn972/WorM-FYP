from src.data.generator import Generator, Colors
from enum import Enum
from typing import List, Dict
import math
import random
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from src.tasks import Tasks

class Variant(Enum):

    RECALL = "recall"
    RECOGNITION = "recognition"

class RecognitionGroundTruth(Enum):

    LEFT = 0
    RIGHT = 1


class Config:
    
    ACTIVE_CELL_COLOR = Colors.GREEN.value
    INACTIVE_CELL_COLOR = Colors.WHITE.value

    CELL_OUTLINE_COLOR = Colors.BLACK.value
    PROBE_LINE_COLOR = Colors.BLACK.value

    MEMORY_STIM_PADDING = 9
    PROBE_STIM_PADDING = 3

    MEMORY_CELL_RADIUS = 3
    PROBE_CELL_RADIUS = 1

    PROBE_LINE_WIDTH = 1


class Visual_Serial_Task_Generator(Generator):

    def __init__(self,
                 grid_size:int, img_size:int,
                 variant:Variant,
                 list_length_options:List[int],
                 held_out_list_length_options:List[int],
                 distractor_diff_options: List[int],
                 num_samples:int,
                 data_dir:str="",
                 save:bool = True,
                 generate_trials = True
                ):
        
        self.config = Config()
        
        self.grid_size = grid_size
        self.img_size = (img_size,img_size)

        self.variant = variant
        self.task_name = Tasks.VISUAL_SERIAL_RECALL if self.variant == Variant.RECALL else Tasks.VISUAL_SERIAL_RECOGNITION

        self.list_length_options = list_length_options
        self.distractor_diff_options = distractor_diff_options

        self.held_out_list_length_options = held_out_list_length_options
        self.held_out_distractor_diff_options = self.held_out_distractor_diff_options

        self.train_num_samples = num_samples 
        self.test_num_samples = (int) (num_samples * 0.1)
        self.gen_test_num_samples = (int) (num_samples * 0.1)

        self.data_dir = data_dir
        self.save = save

        # Assertions
        assert self.train_num_samples % len(list_length_options) == 0
        assert self.test_num_samples % len(list_length_options) == 0
        max_list_length = max(self.list_length_options)

        assert (
            (max_list_length % 2 == 0 or 
             max_list_length == int(math.sqrt(max_list_length)) ** 2)
             and (max_list_length >= 4)
        )

        assert all(
            distractor_diff % 2 == 0
            for distractor_diff in self.distractor_diff_options
        )

        if self.held_out_list_length_options:
            assert self.gen_test_num_samples % len(held_out_list_length_options) == 0
            held_out_max_list_length = max(self.held_out_list_length_options)
        
            assert (
                (held_out_max_list_length % 2 == 0 or 
                held_out_max_list_length == int(math.sqrt(max_list_length)) ** 2)
                and (held_out_max_list_length >= 4)
            )

        if self.held_out_distractor_diff_options:
            assert all(
                distractor_diff % 2 == 0
                for distractor_diff in self.held_out_distractor_diff_options
            )

        if self.save:
            self.train_dir, self.test_dir, self.gen_test_dir = self._init_dirs(self.data_dir, f"{self.task_name.name.lower()}")


        if generate_trials:
            self.generate_trials()

    
    def generate_trials(self):
        self._log("Generating trials", variant=self.variant)        

        trials_dict = {"train": [], "test": [], "gen_test": []}

        for trial_type in self.trial_types:
            self._log(f"Split: {trial_type}", variant=self.variant)
            
            # trial_length_options is dependant on the trial type
            if trial_type in ["train", "test"]:

                list_length_options = self.list_length_options
                distractor_diff_options = self.distractor_diff_options

            else:
                list_length_options = self.held_out_list_length_options
                distractor_diff_options = self.held_out_distractor_diff_options

            # Each Combination is (set_size, presentation_time)
            total_trial_samples = getattr(self, f"{trial_type}_num_samples")
            total_trial_samples_per_combination = total_trial_samples // len(list_length_options)
            
            cur_total_trial_samples = 0
            cur_trial_samples_per_combination = {}

            # Loop over each combination
            for list_length in tqdm(list_length_options, leave=False, desc=f"{trial_type} | list_length"):

                combination = list_length
                cur_trial_samples_per_combination[combination] = 0

                while(cur_trial_samples_per_combination[combination] < total_trial_samples_per_combination):
                    
                    num_grid_cells = self.grid_size ** 2
                    grid_cells = range(num_grid_cells)

                    # -- Memory Stim --
                    memory_items = []

                    # For each memory stim in the sequence
                    # Half the grid cells are active 
                    for _ in range(list_length):
                        active_cells = random.sample(grid_cells, num_grid_cells // 2)
                        active_cells.sort()

                        # make sure there are no duplicates in the 
                        while active_cells in memory_items:
                            active_cells = random.sample(grid_cells, num_grid_cells // 2)
                            active_cells.sort()

                        memory_items.append(active_cells)


                    # Now its time logic based on variant 
                    if self.variant == Variant.RECALL:
                        
                        num_probe_stims = 1

                        # We choose a probe position of each memory item in the grid 
                        probe_grid_size = max(list_length_options)
                        recall_gt = random.sample(range(probe_grid_size), list_length)

                        probe_dict = {
                            "recall_gt": recall_gt
                        }


                    elif self.variant == Variant.RECOGNITION:
                        
                        num_probe_stims = list_length

                        # Randomly choos a distractor diff
                        distractor_diff = random.choice(distractor_diff_options)
                        diff_sample_num = distractor_diff // 2

                        distractor_items = []
                        recog_gt = []

                        for item in memory_items:

                            inactive_cells = [cell for cell in grid_cells if cell not in item]
                            
                            # sample one cell to make inactive
                            new_inactive_cells = random.sample(item, diff_sample_num)

                            # sample one cell to make active 
                            new_active_cells = random.sample(inactive_cells, diff_sample_num)

                            distractor_active_cells = [cell for cell in item if cell not in new_inactive_cells]
                            distractor_active_cells.extend(new_active_cells)

                            # Choose ground truth 
                            gt = random.choice(list(RecognitionGroundTruth))
                            
                            distractor_items.append(distractor_active_cells)
                            recog_gt.append(gt.value)

                        probe_dict = {
                            "distractor_diff": distractor_diff,
                            "distractor_items": distractor_items,
                            "recog_gt": recog_gt,
                            "recog_gt_values": ["left" if gt == 0 else "right" for gt in recog_gt],
                        }

                    else:
                        raise ValueError(f"Variant {self.variant.name} not implemented")

                    # -- Generate File Names --
                    id = cur_total_trial_samples
                    memory_stim_fnames = [
                        f"{trial_type}_{str(id).zfill(6)}_memory{str(idx).zfill(2)}.png" for idx in range(list_length)
                    ]

                    probe_stim_fnames = [
                            f"{trial_type}_{str(id).zfill(6)}_probe{str(idx).zfill(2)}.png" for idx in range(num_probe_stims)
                        ]


                    # Adding the trial_data to trials
                    trials_dict[trial_type].append({
                        "trial_type": trial_type,
                        "trial_id": id, 
                        
                        "grid_size": self.grid_size,

                        "list_length": list_length,
                        "memory_items": memory_items,

                        **probe_dict,

                        "memory_stim_fnames": memory_stim_fnames,
                        "probe_stim_fnames": probe_stim_fnames,
                    })

                    # Update Counters
                    cur_total_trial_samples += 1
                    cur_trial_samples_per_combination[combination] += 1

        if self.save:
            self._save_trial_json(trials_dict)

        self._draw_trial_stims(trials_dict)

        self._log_summary(trials_dict)

        return trials_dict
    
    def _draw_trial_stims(self, trials):

        trials_dict = {"train": [], "test": [], "gen_test": []}

        for trial_type in self.trial_types:

            trials_list = trials[trial_type]

            # We go over each trial 
            for trial in trials_list:

                # Draw the memory stim 
                memory_stims = self._draw_memory_stims(trial)

                # Draw the probe stim 
                probe_stims = self._draw_probe_stims(trial)

                # Saving Logic 
                if self.save:
                    self._save_stim(trial_type, memory_stims, trial["memory_stim_fnames"])
                    self._save_stim(trial_type, probe_stims, trial["probe_stim_fnames"])

    
    def _draw_memory_stims(self, trial):

        memory_items = trial["memory_items"]
        
        stims = []

        for memory_item in memory_items:

            stim, draw = self._init_stim(self.img_config.PRESENTATION_OUTLINE_COLOR)

            # Calculate the grid_size in px
            # We need to subtract the padding 
            grid_size_px = self.img_size[0] - self.config.MEMORY_STIM_PADDING*2
            
            # Calculate the size of each grid_cell in px
            grid_cell_size_px = grid_size_px // self.grid_size

            offset = (self.img_size[0] - grid_size_px) // 2

            # Draw the cells
            for cell in range(self.grid_size ** 2):
                cell_color = self.config.ACTIVE_CELL_COLOR if cell in memory_item else self.config.INACTIVE_CELL_COLOR
                self._draw_grid_cell(draw, 
                                     cell,offset, offset,
                                     grid_cell_size_px, self.config.MEMORY_CELL_RADIUS,
                                     cell_color, self.config.CELL_OUTLINE_COLOR)
  
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


    def _draw_probe_stims(self, trial):

        if self.variant == Variant.RECALL:
            return self.draw_recall_probe_stims(trial)

        if self.variant == Variant.RECOGNITION:
            return self.draw_recog_probe_stims(trial)


    def draw_recog_probe_stims(self, trial):

        memory_items = trial["memory_items"]

        recog_gt = trial["recog_gt"]
        distractor_items = trial["distractor_items"]

        stims = []

        for gt,true_memory,distractor in zip(recog_gt, memory_items, distractor_items):

            if gt == 0:  # Left 
                left_stim = true_memory
                right_stim = distractor

            else:
                left_stim = distractor
                right_stim = true_memory


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

                
                # Draw the cells
                for cell in range(self.grid_size ** 2):
                    cell_color = self.config.ACTIVE_CELL_COLOR if cell in cur_stim else self.config.INACTIVE_CELL_COLOR
                    self._draw_grid_cell(draw, 
                                        cell,x_offset, y_offset,
                                        grid_cell_size_px, self.config.MEMORY_CELL_RADIUS,
                                        cell_color, self.config.CELL_OUTLINE_COLOR)
    
            draw.line([half_width, 20, half_width, self.img_size[1]-20],
                    fill=self.config.PROBE_LINE_COLOR,
                    width=self.config.PROBE_LINE_WIDTH)
            

            # Draw Right Grid 
            stims.append(stim)

        return stims
    
    def draw_recall_probe_stims(self, trial):

        memory_items = trial["memory_items"]
        recall_gt = trial["recall_gt"]
        trial_type = trial.get("trial_type", "train")

        # Match how you sampled recall_gt in generate_trials:
        # train/val uses self.list_length_options, test uses held_out_list_length_options
        if trial_type in ["train", "test"]:
            max_list_length = max(self.list_length_options)
        else:
            max_list_length = max(self.held_out_list_length_options)

        # Decide slot layout:
        # - perfect square => sqrt x sqrt
        # - else (even) => 2 x (N/2)
        sqrt_n = int(math.sqrt(max_list_length))
        if sqrt_n * sqrt_n == max_list_length:
            num_rows = sqrt_n
            num_cols = sqrt_n
        else:
            num_rows = 2
            num_cols = max_list_length // 2

        stim, draw = self._init_stim(self.img_config.PROBE_OUTLINE_COLOR)

        W, H = self.img_size
        slot_w = W // num_cols
        slot_h = H // num_rows

        # Draw slot grid lines
        for r in range(1, num_rows):
            y = r * slot_h
            draw.line([0, y, W, y],
                    fill=self.config.PROBE_LINE_COLOR,
                    width=self.config.PROBE_LINE_WIDTH)

        for c in range(1, num_cols):
            x = c * slot_w
            draw.line([x, 0, x, H],
                    fill=self.config.PROBE_LINE_COLOR,
                    width=self.config.PROBE_LINE_WIDTH)

        # Mini-grid sizing inside a slot (square, centered)
        pad = self.config.PROBE_STIM_PADDING
        mini_px = min(slot_w, slot_h) - 2 * pad

        grid_cell_size_px = mini_px // self.grid_size
        actual_grid_width = self.grid_size * grid_cell_size_px

        internal_margin_x = (mini_px - actual_grid_width) // 2
        internal_margin_y = (mini_px - actual_grid_width) // 2

        # Place each memory item into its assigned slot
        for item_idx, slot_idx in enumerate(recall_gt):
            row = slot_idx // num_cols
            col = slot_idx % num_cols

            slot_x0 = col * slot_w
            slot_y0 = row * slot_h

            x_offset = slot_x0 + pad + internal_margin_x
            y_offset = slot_y0 + pad + internal_margin_y

            active_cells = memory_items[item_idx]

            for cell in range(self.grid_size ** 2):
                cell_color = (self.config.ACTIVE_CELL_COLOR
                            if cell in active_cells
                            else self.config.INACTIVE_CELL_COLOR)

                self._draw_grid_cell(
                    draw,
                    cell,
                    x_offset, y_offset,
                    grid_cell_size_px,
                    self.config.PROBE_CELL_RADIUS,
                    cell_color,
                    self.config.CELL_OUTLINE_COLOR
                )

        return [stim]

        

                    



