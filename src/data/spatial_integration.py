from src.data.generator import Generator, Colors
from typing import List, Dict
from enum import Enum
import random
from copy import deepcopy
from tqdm.auto import tqdm

from src.tasks import Tasks


class GroundTruth(Enum):

    MATCH = 1
    MISMATCH = 0

class Direction(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"

    UP_LEFT = "up-left"
    UP_RIGHT = "up-right"
    DOWN_LEFT = "down-left"
    DOWN_RIGHT = "down-right"

class Config:
    
    LINE_WIDTH = 3
    BORDER_PADDING = 5

    PROBE_BORDER_WIDTH = 3
    STIM_IMG_PADDING = 10

    LINE_COLOR = Colors.GREEN.value


    DIRECTION_MATH = {
        Direction.UP: (-1,0),
        Direction.DOWN: (1,0),
        Direction.LEFT: (0,-1),
        Direction.RIGHT: (0,1),
        
        Direction.UP_LEFT: (-1,-1),
        Direction.UP_RIGHT: (-1,1),
        Direction.DOWN_LEFT: (1,-1),
        Direction.DOWN_RIGHT: (1,1),
    }

class Spatial_Integration_Generator(Generator):

    def __init__(self,
                 num_samples:int,
                 img_size:int,
                 grid_size_options:List[int], held_out_grid_size_options:List[int],
                 pattern_size_options:List[int], held_out_pattern_size_options:List[int],
                 distractor_difference_options:List[int], held_out_distractor_difference_options:List[int],
                 part_size_options:List[int], held_out_part_size_options:List[int],
                 max_retries: int,
                 data_dir:str="",save:bool=True, generate_trials=True):
        
        self.config = Config()

        self.grid_size_options = grid_size_options
        self.held_out_grid_size_options = held_out_grid_size_options

        self.img_size = (img_size,img_size)

        self.pattern_size_options = pattern_size_options
        self.held_out_pattern_size_options = held_out_pattern_size_options

        self.distractor_difference_options = distractor_difference_options
        self.held_out_distractor_difference_options = held_out_distractor_difference_options

        self.part_size_options = part_size_options
        self.held_out_part_size_options = held_out_part_size_options

        self.max_retries = max_retries

        self.train_num_samples = num_samples 
        self.test_num_samples = (int) (num_samples * 0.1)
        self.gen_test_num_samples = (int) (num_samples * 0.1)

        self.data_dir = data_dir
        self.save = save

        # Assertions

        assert self.train_num_samples % len(part_size_options) == 0
        assert self.test_num_samples % len(part_size_options)== 0
        assert self.gen_test_num_samples % len(held_out_part_size_options) == 0

        assert self.max_retries > max(pattern_size_options)
            
        assert all([pattern_size % part_size == 0
                    for pattern_size in pattern_size_options 
                    for part_size in part_size_options])

        assert all([
            pattern_size < grid_size ** 2 
            for pattern_size in pattern_size_options
            for grid_size in grid_size_options
        ])

        assert all([
            pattern_size < grid_size ** 2 
            for pattern_size in held_out_pattern_size_options
            for grid_size in held_out_grid_size_options
        ])

        if self.save:
            self.train_dir, self.test_dir, self.gen_test_dir = self._init_dirs(self.data_dir, Tasks.SPATIAL_INTEGRATION.name.lower())

        if generate_trials:
            self.generate_trials()

    def generate_trials(self):

        self._log("Generating trials")        
        
        trials_dict = {"train": [], "test": [], "gen_test": []}

        for trial_type in self.trial_types:

            self._log(f"Split: {trial_type}")
            
            # The following variables are dependant on the trial type
            if trial_type in ["train", "test"]:

                part_size_options = self.part_size_options
                grid_size_options = self.grid_size_options
                pattern_size_options = self.pattern_size_options
                distractor_diff_options = self.distractor_difference_options

            else:
                part_size_options = self.held_out_part_size_options
                grid_size_options = self.held_out_grid_size_options
                pattern_size_options = self.held_out_pattern_size_options
                distractor_diff_options = self.held_out_distractor_difference_options

            # Each Combination is (part_size)
            total_trial_samples = getattr(self, f"{trial_type}_num_samples")
            total_trial_samples_per_combination = total_trial_samples // (len(part_size_options))
            
            cur_total_trial_samples = 0
            cur_trial_samples_per_combination = {}

            # Loop over each combination
            for part_size in tqdm(part_size_options, leave=False, desc=f"{trial_type} | part_size"):

                combination = part_size
                cur_trial_samples_per_combination[combination] = 0

                while(cur_trial_samples_per_combination[combination] < total_trial_samples_per_combination):

                    # Choose a random grid size
                    grid_size = random.choice(grid_size_options)

                    # Choose a random pattern size 
                    pattern_size = random.choice(pattern_size_options)

                    # Time to generate the pattern 
                    pattern = self._generate_pattern(grid_size, pattern_size)


                    # Split the memory pattern into contiguous chunks
                    num_parts, parts = self._split_pattern(pattern, pattern_size, part_size)
                    random.shuffle(parts)

                    # Decide the Ground Truth and Generate the distractor pattern accordingly
                    gt = random.choice(list(GroundTruth))

                    if gt == GroundTruth.MISMATCH:

                        distractor_diff = random.choice(distractor_diff_options)
                        probe_pattern = self._generate_distractor_pattern(grid_size, pattern_size, pattern, distractor_diff)

                    else:
                        probe_pattern = pattern
                        distractor_diff = None


                # -- Generate File names --
                    id = cur_total_trial_samples
                    memory_stim_fnames = [
                        f"{trial_type}_{str(id).zfill(6)}_memory{str(idx).zfill(2)}.png" for idx in range(num_parts)
                    ]

                    probe_stim_fnames = [
                        f"{trial_type}_{str(id).zfill(6)}_probe{str(idx).zfill(2)}.png" for idx in range(1)
                    ]

                    # Adding the trial_data to trials
                    trials_dict[trial_type].append({
                        "trial_type": trial_type,
                        "trial_id": id, 
                        
                        "grid_size": grid_size,

                        "pattern_size": pattern_size,
                        "part_size": part_size,
                        "num_parts": num_parts,

                        "memory_pattern": pattern,
                        "memory_pattern_parts": parts,

                        "gt": gt.value,
                        "distractor_diff": distractor_diff,
                        "probe_pattern": probe_pattern,

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



    def _init_pattern(self, grid_size):

        total_rows = grid_size
        total_cols = grid_size

        # Choose a random start point
        start_coords = random.choice(range(total_rows)), random.choice(range(total_cols))

        # Add the start coords to visited
        visited = set()
        visited.add(start_coords)

        # Add the start coords to the pattern
        pattern = [start_coords]

        return visited, pattern

    def _generate_pattern(self, grid_size, pattern_size):
        
        visited, pattern = self._init_pattern(grid_size)

        fail_count = 0

        while len(pattern) < pattern_size + 1:

            cur_row = pattern[-1][0]
            cur_col = pattern[-1][1]

            next_coords = self._sample_next_point(cur_row,cur_col)

            # Check for Validity and add to Pattern if not visited
            if not self._valid_coords(next_coords, grid_size, visited):

                fail_count += 1
                    
            else:   
                fail_count = 0
                pattern.append(next_coords)
                visited.add(next_coords)

            if fail_count == self.max_retries:
                fail_count = 0
                visited, pattern = self._init_pattern(grid_size)

        return pattern
    
    
    def _split_pattern(self, pattern, pattern_size, part_size):

        num_parts = pattern_size // part_size
        
        parts = []

        cur_idx = 0
        count = 0

        while count < num_parts:

            end_idx = cur_idx + part_size
            parts.append(pattern[cur_idx:end_idx + 1])
            cur_idx = end_idx

            count += 1

        return num_parts, parts
    
    def _generate_distractor_pattern(self, grid_size, pattern_size, pattern, distractor_diff):

        # based on the distractor diff 
        # remove the last distractor diff points in the pattern

        new_pattern = pattern[:-distractor_diff].copy()

        fail_count = 0
        visited = set(new_pattern)

        while len(new_pattern) < pattern_size + 1:

            cur_row = new_pattern[-1][0]
            cur_col = new_pattern[-1][1]

            next_coords = self._sample_next_point(cur_row,cur_col)

            # Check for Validity and add to Pattern if not visited
            if not self._valid_coords(next_coords, grid_size, visited):
                fail_count += 1
                    
            else:   
                fail_count = 0
                new_pattern.append(next_coords)
                visited.add(next_coords)

            if fail_count >= self.max_retries:
                fail_count = 0
                new_pattern = pattern[:-distractor_diff].copy()
                visited = set(new_pattern)


        return new_pattern
    
    def _sample_next_point(self, cur_row, cur_col):
        update_dir = random.choice(list(Direction))
        row_diff, col_diff = self.config.DIRECTION_MATH[update_dir]
            
        next_row = cur_row + row_diff
        next_col = cur_col + col_diff

        next_coords = (next_row, next_col)
        
        return next_coords
    

    def _valid_coords(self, coords, grid_size, visited):

        next_row = coords[0]
        next_col = coords[1]

        return (0 <= next_row < grid_size
            and 0 <= next_col < grid_size
            and coords not in visited)
    

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

        grid_size = trial["grid_size"]

        memory_pattern_parts = trial["memory_pattern_parts"]

        memory_stims = []

        for pattern_segment in memory_pattern_parts:

            stim, draw = self._init_stim(self.img_config.PRESENTATION_OUTLINE_COLOR)
        
            self._draw_pattern(draw, pattern_segment, grid_size, self.config.STIM_IMG_PADDING)

            memory_stims.append(stim)

        return memory_stims
    

    def _draw_probe_stims(self, trial):

        grid_size = trial["grid_size"]
        probe_pattern = trial["probe_pattern"]

        probe_stims = []

        stim, draw = self._init_stim(self.img_config.PROBE_OUTLINE_COLOR)
        self._draw_pattern(draw, probe_pattern, grid_size, self.config.STIM_IMG_PADDING)


        probe_stims.append(stim)


        return probe_stims

        
    
    def _draw_pattern(self, draw, pattern_segment, grid_size, padding):

        grid_px = self.img_size[0] - 2*padding
        cell_px = grid_px // grid_size

        x_px = lambda col : col * cell_px + padding
        y_px = lambda row: row * cell_px + padding

        for i in range(len(pattern_segment) - 1):
            p1 = pattern_segment[i]
            p2 = pattern_segment[i + 1]

            x1, y1 = x_px(p1[1]), y_px(p1[0])
            x2, y2 = x_px(p2[1]), y_px(p2[0])

            draw.line((x1,y1,x2,y2), width=self.config.LINE_WIDTH, fill=self.config.LINE_COLOR)




        






            

            


                        

            

              

            
                



