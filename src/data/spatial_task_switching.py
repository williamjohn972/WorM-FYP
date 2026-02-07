from src.data.generator import Generator, Colors
from PIL import Image, ImageDraw
from typing import List, Dict
from enum import Enum
import random
from tqdm.auto import tqdm

from src.tasks import Tasks

class Variant(Enum):

    CUED = "cued"      # Cue Frame appear (The model is told explicitly to switch)
    ALTERNATE = "alternate" # Task alternates in a fixed pattern 


class TaskGroundTruth(Enum):

    CUE_UP_DOWN = "cue_up_down"
    CUE_LEFT_RIGHT = "cue_left_right"
    
    UP_DOWN = "up_down"
    LEFT_RIGHT = "left_right"

class GroundTruth(Enum):

    CUE_FRAME = 2

    TOP = 0
    LEFT = 0
    
    BOTTOM = 1
    RIGHT = 1

class MarkerLocation(Enum):

    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"

    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


class Config:

    SWITCH_CUE_TO_TASK_PROB_THRESHOLD = 0.8

    CUE_TO_TASK_MAP = {
        TaskGroundTruth.CUE_UP_DOWN: TaskGroundTruth.UP_DOWN,
        TaskGroundTruth.CUE_LEFT_RIGHT: TaskGroundTruth.LEFT_RIGHT
    }

    TASK_TO_GROUND_TRUTH_MAP = {
        TaskGroundTruth.CUE_UP_DOWN: [GroundTruth.CUE_FRAME],
        TaskGroundTruth.CUE_LEFT_RIGHT: [GroundTruth.CUE_FRAME],
        
        TaskGroundTruth.UP_DOWN: [GroundTruth.TOP, GroundTruth.BOTTOM],
        TaskGroundTruth.LEFT_RIGHT: [GroundTruth.LEFT, GroundTruth.RIGHT],
    }

    GT_TO_MARKER_LOCATION_MAP = {
        GroundTruth.TOP : [MarkerLocation.TOP_LEFT, MarkerLocation.TOP_RIGHT],
        GroundTruth.BOTTOM : [MarkerLocation.BOTTOM_LEFT, MarkerLocation.BOTTOM_RIGHT],
        GroundTruth.LEFT : [MarkerLocation.TOP_LEFT, MarkerLocation.BOTTOM_LEFT],
        GroundTruth.RIGHT: [MarkerLocation.TOP_RIGHT, MarkerLocation.BOTTOM_RIGHT],
    }

    MARKER_LOCATION_TO_GRID_CELL_MAP = {
        MarkerLocation.TOP_LEFT.value: 0,
        MarkerLocation.TOP_RIGHT.value: 1,

        MarkerLocation.BOTTOM_LEFT.value: 2,
        MarkerLocation.BOTTOM_RIGHT.value: 3,
    }

    GRID_PADDING = 20
    GRID_LINE_WIDTH = 2

    MARKER_PADDING = 8
    RECT_RADIUS = 2
    MARKER_RECT_RADIUS = 1

    CUE_LINE_PADDING = 4

    GRID_FILL_COLOR = Colors.WHITE.value
    GRID_OUTLINE_COLOR = Colors.BLACK.value
    MARKER_COLOR = Colors.RED.value


class Spatial_Task_Switching_Generator(Generator):

    def __init__(self, 
                 img_size: int, 
                 trial_length_options: List[int], held_out_trial_length_options: List[int],
                 num_samples:int, 
                 variant: Variant,
                 data_dir:str = "", save: bool = True, generate_trials = True):

        self.config = Config()
        
        self.img_size = (img_size,img_size)

        self.variant = variant

        self.cue_list = [TaskGroundTruth.CUE_LEFT_RIGHT, TaskGroundTruth.CUE_UP_DOWN]

        self.trial_length_options = trial_length_options
        self.held_out_trial_length_options = held_out_trial_length_options

        self.train_num_samples = num_samples 
        self.test_num_samples = (int) (num_samples * 0.1)
        self.gen_test_num_samples = (int) (num_samples * 0.1)

        self.data_dir = data_dir
        self.save = save

        # Assertions
        assert self.train_num_samples % len(trial_length_options) == 0
        assert self.test_num_samples % len(trial_length_options) == 0
        assert self.gen_test_num_samples % len(held_out_trial_length_options) == 0
            
        if self.save:
            self.train_dir, self.test_dir, self.gen_test_dir = self._init_dirs(self.data_dir, f"{Tasks.SPATIAL_TASK_SWITCHING.name.lower()} - {variant.value.lower()}")

        if generate_trials:
            self.generate_trials()


    def generate_trials(self):
        self._log("Generating trials")        
        
        trials_dict = {"train": [], "test": [], "gen_test": []}

        for trial_type in self.trial_types:
            self._log(f"Split: {trial_type}")
            
            # trial_length_options is dependant on the trial type
            if trial_type in ["train", "test"]:

                trial_length_options = self.trial_length_options

            else:
                trial_length_options = self.held_out_trial_length_options

            # Each Combination is (set_size, presentation_time)
            total_trial_samples = getattr(self, f"{trial_type}_num_samples")
            total_trial_samples_per_combination = total_trial_samples // len(trial_length_options)
            
            cur_total_trial_samples = 0
            cur_trial_samples_per_combination = {}

            # Loop over each combination
            for trial_length in tqdm(trial_length_options, leave=False, desc=f"{trial_type} | trial_length"):

                combination = trial_length
                cur_trial_samples_per_combination[combination] = 0

                while(cur_trial_samples_per_combination[combination] < total_trial_samples_per_combination):
                    
                    task_order, gt_order, marker_locations = self._generate_task_seq(trial_length)

                    # -- Generate File names --
                    id = cur_total_trial_samples
                    
                    stim_fnames = [
                        f"{trial_type}_{str(id).zfill(6)}_task{str(idx).zfill(2)}.png" for idx in range(trial_length)
                    ]

                    # Adding the trial_data to trials
                    trials_dict[trial_type].append({
                        "trial_type": trial_type,
                        "trial_id": id, 

                        "trial_length": trial_length,

                        "variant": self.variant.value,

                        "task_order": [task.value for task in task_order],
                        "gt_order": [gt.value for gt in gt_order],
                        "marker_locations" : [loc.value if loc !=None else None for loc in marker_locations],

                        "stim_fnames": stim_fnames,
                    })

                    # Update Counters
                    cur_total_trial_samples += 1
                    cur_trial_samples_per_combination[combination] += 1

        if self.save:
            self._save_trial_json(trials_dict)

        self._draw_trial_stims(trials_dict)

        self._log_summary(trials_dict)

        return trials_dict

    def _generate_task_seq(self, trial_length):

        cue = random.choice(self.cue_list)
        task_order = [cue]
        gt_order = [random.choice(self.config.TASK_TO_GROUND_TRUTH_MAP[cue])]
        marker_location = [None]

        while len(task_order) < trial_length:
            
            # The previous task could be a cue or a normal task
            prev_task = task_order[-1]

            # If the prev_task was a cue then the follow task is a normal task
            if prev_task in self.cue_list:
                cur_task = self.config.CUE_TO_TASK_MAP[prev_task]

            # If the prev_task was a normal task then the following task could be a normal task or a cue
            else:
                candidate_cue = random.choice(self.cue_list)
                switch_probability = random.random()
                
                cur_task = candidate_cue if switch_probability > self.config.SWITCH_CUE_TO_TASK_PROB_THRESHOLD else prev_task
        

            # gt depends on the cur_task
            cur_gt = random.choice(self.config.TASK_TO_GROUND_TRUTH_MAP[cur_task])

            # marker position depends on gt
            cur_marker_location = random.choice(self.config.GT_TO_MARKER_LOCATION_MAP[cur_gt]) if cur_gt != GroundTruth.CUE_FRAME else None

            # Append the task and the gt
            task_order.append(cur_task)
            gt_order.append(cur_gt)
            marker_location.append(cur_marker_location)

        return task_order, gt_order, marker_location


    def _draw_trial_stims(self, trials):

        trials_dict = {"train": [], "test": [], "gen_test": []}

        for trial_type in self.trial_types:

            trials_list = trials[trial_type]

            # We go over each trial 
            for trial in trials_list:

                # Draw the memory stim 
                trial_stims = self._draw_stims(trial)

                # Saving Logic 
                if self.save:
                    self._save_stim(trial_type, trial_stims, trial["stim_fnames"])


    def _draw_stims(self, trial):

        stims = []

        trial_length = trial["trial_length"]
        
        task_order = trial["task_order"]
        gt_order = trial["gt_order"]
        marker_locations = trial["marker_locations"]


        for idx in range(trial_length):

            cur_task = task_order[idx]

            if cur_task in [cue.value for cue in self.cue_list]:
                stim, draw = self._init_stim(self.img_config.TASK_SWITCH_OUTLINE_COLOR)
                self._draw_task_switch_stim(draw, cur_task)
                
            else: 
                stim, draw = self._init_stim(self.img_config.PROBE_OUTLINE_COLOR)

                cur_marker_loc = marker_locations[idx]
                self._draw_probe_stim(draw, cur_marker_loc)
        
            stims.append(stim)

        return stims
    

    def _draw_task_switch_stim(self, draw, cue):
        
        self._draw_grid(draw)

        img_size = self.img_size[0]
        mid_x = img_size // 2
        mid_y = img_size // 2

        grid_padding = self.config.GRID_PADDING
        cue_line_padding = self.config.CUE_LINE_PADDING


        if cue == TaskGroundTruth.CUE_UP_DOWN.value:
            
            draw.line(
                [ mid_x, cue_line_padding, mid_x, grid_padding - cue_line_padding],
                fill=self.config.GRID_OUTLINE_COLOR,
                width=self.config.GRID_LINE_WIDTH,
            )

            draw.line(
                [mid_x,img_size - grid_padding + cue_line_padding, mid_x,img_size - cue_line_padding],
                fill=self.config.GRID_OUTLINE_COLOR,
                width=self.config.GRID_LINE_WIDTH,
            )


        elif cue == TaskGroundTruth.CUE_LEFT_RIGHT.value:
            # Left line
            draw.line(
                [cue_line_padding,mid_y,grid_padding - cue_line_padding,mid_y],
                fill=self.config.GRID_OUTLINE_COLOR,
                width=self.config.GRID_LINE_WIDTH,
            )

            # Right line
            draw.line(
                [img_size - grid_padding + cue_line_padding,mid_y,img_size - cue_line_padding,mid_y],
                fill=self.config.GRID_OUTLINE_COLOR,
                width=self.config.GRID_LINE_WIDTH,
            )




    
    def _draw_probe_stim(self, draw, marker_loc):

        grid_cell_size_px = self._draw_grid(draw)
        
        grid_padding = self.config.GRID_PADDING
        marker_padding = self.config.MARKER_PADDING

        marker_size = grid_cell_size_px - 2 * self.config.MARKER_PADDING
        marker_grid_cell = self.config.MARKER_LOCATION_TO_GRID_CELL_MAP[marker_loc]

        row = marker_grid_cell // 2
        col = marker_grid_cell % 2

        cell_x = grid_padding + col * grid_cell_size_px
        cell_y = grid_padding + row * grid_cell_size_px

        # Marker position inside cell
        marker_x1 = cell_x + marker_padding
        marker_y1 = cell_y + marker_padding
        marker_x2 = marker_x1 + marker_size
        marker_y2 = marker_y1 + marker_size

        # Draw marker
        draw.rounded_rectangle(
            [marker_x1, marker_y1, marker_x2, marker_y2],
            radius=self.config.MARKER_RECT_RADIUS,
            fill=self.config.MARKER_COLOR,
        )


    def _draw_grid(self, draw):

        grid_size_px = self.img_size[0] - 2 * self.config.GRID_PADDING
        grid_cell_size_px = grid_size_px // 2

        grid_padding = self.config.GRID_PADDING

        for cell_idx in range(4):
            
            row = cell_idx // 2
            col = cell_idx % 2

            x1 = grid_padding + col * grid_cell_size_px
            y1 = grid_padding + row * grid_cell_size_px

            x2 = x1 + grid_cell_size_px
            y2 = y1 + grid_cell_size_px

            draw.rectangle(
                [x1,y1,x2,y2],
                fill= self.config.GRID_FILL_COLOR,
                outline = self.config.GRID_OUTLINE_COLOR
            )

        return grid_cell_size_px




        

        


                        