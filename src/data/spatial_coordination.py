from src.data.generator import Generator, Colors
from enum import Enum
from typing import List, Dict
import random
from tqdm.auto import tqdm

from src.tasks import Tasks

class GroundTruth(Enum):
    SYMETRIC = 1 
    ASYMETRIC = 0 

class Config:

    ACTIVE_CELL_COLOR = Colors.GREEN.value
    INACTIVE_CELL_COLOR = Colors.WHITE.value

    GRID_OUTLINE_COLOR = Colors.BLACK.value

    GRID_PADDING = 5


class Spatial_Coordination_Generator(Generator):

    def __init__(self,
                 grid_size:int, img_size:int,
                 num_samples:int,
                 list_length_options:List[int], held_out_list_length_options:List[int],
                 symetry_offset_options:List[int], held_out_symetry_offset_options:List[int], 
                 data_dir:str="",save:bool=True, generate_trials=True):
        
        self.config = Config()

        self.grid_size = grid_size
        self.img_size = (img_size,img_size)

        self.list_length_options = list_length_options
        self.held_out_list_length_options = held_out_list_length_options

        self.symetry_offset_options = symetry_offset_options
        self.held_out_symetry_offset_options = held_out_symetry_offset_options

        self.train_num_samples = num_samples 
        self.test_num_samples = (int) (num_samples * 0.1)
        self.gen_test_num_samples = (int) (num_samples * 0.1)

        self.data_dir = data_dir
        self.save = save

        # Assertions
        assert self.grid_size % 2 == 0 # Grid size must be even for left right symetry

        assert all(list_length % 2 == 0 for list_length in self.list_length_options) # List Lengths must be even
        
        if self.held_out_list_length_options:
            assert all(list_length % 2 == 0 for list_length in self.held_out_list_length_options) # List Lengths must be even
            assert max(self.held_out_list_length_options) <= self.grid_size**2 # List length cannot exceed number of grid cells

        assert max(self.list_length_options) <= self.grid_size**2 # List length cannot exceed number of grid cells

        assert all(opt % 2== 0 for opt in symetry_offset_options)

        if self.held_out_symetry_offset_options:
            assert all(opt % 2== 0 for opt in held_out_symetry_offset_options)

        
        for sym_off in symetry_offset_options:
            for list_length in list_length_options:
                assert(sym_off <= list_length)


        if self.held_out_symetry_offset_options or self.held_out_list_length_options:
            for sym_off in self.held_out_symetry_offset_options:
                for list_length in held_out_list_length_options:
                    assert(sym_off <= list_length)

        assert self.train_num_samples % len(list_length_options) == 0
        assert self.test_num_samples % len(list_length_options)== 0

        if self.held_out_list_length_options:
            assert self.gen_test_num_samples % len(held_out_list_length_options) == 0
            
        if self.save:
            self.train_dir, self.test_dir, self.gen_test_dir = self._init_dirs(self.data_dir, Tasks.SPATIAL_COORDINATION.name.lower())

        if generate_trials:
            self.generate_trials()


    def generate_trials(self):

        self._log("Generating trials")        
        
        trials_dict = {"train": [], "test": [], "gen_test": []}

        for trial_type in self.trial_types:
            self._log(f"Split: {trial_type}")
            
            # list_length_options is dependant on the trial type
            if trial_type in ["train", "test"]:

                list_length_options = self.list_length_options
                symetry_offset_options = self.symetry_offset_options

            else:
                list_length_options = self.held_out_list_length_options
                symetry_offset_options = self.symetry_offset_options

            # Each Combination is (list_length)
            total_trial_samples = getattr(self, f"{trial_type}_num_samples")
            total_trial_samples_per_combination = total_trial_samples // max(1,(len(list_length_options)))
            
            cur_total_trial_samples = 0
            cur_trial_samples_per_combination = {}

            # Loop over each combination
            for list_length in tqdm(list_length_options, leave=False, desc=f"{trial_type} | list_length"):

                combination = list_length
                cur_trial_samples_per_combination[combination] = 0

                while(cur_trial_samples_per_combination[combination] < total_trial_samples_per_combination):
                    
                    left_cells = set()
                    right_cells = set()

                    # Make a note of unsampled_right_cells (useful later)
                    unsampled_grid_right_cells = set((row, col) for row in range(self.grid_size) for col in range(self.grid_size//2, self.grid_size))

                    while len(left_cells) < list_length // 2:
                        
                        # We sample a random cell on the left side
                        row = random.choice(range(self.grid_size))
                        col = random.choice(range(self.grid_size//2))

                        left_cell = (row,col)

                        # right mirror cell calculation
                        # the row is the same 
                        # col --> grid_size - 1 - col
                        if left_cell not in left_cells:
                            right_cell = (row, self.grid_size-1 - col)

                            left_cells.add(left_cell)
                            right_cells.add(right_cell)
                            unsampled_grid_right_cells.remove(right_cell)

                    # Choose the Ground truth
                    gt = random.choice(list(GroundTruth))

                    if(gt == GroundTruth.ASYMETRIC): 
                        
                        # Sample a symety offset 
                        symetry_offset = random.choice(symetry_offset_options)
                        num_broken_pairs = symetry_offset//2
                        
                        # Replace Mirror Right Cells with other random Unsampled Right Cells
                        right_cells_to_replace = random.sample(list(right_cells), num_broken_pairs)
                        new_right_cells = random.sample(list(unsampled_grid_right_cells), num_broken_pairs)

                        for right_cell in right_cells_to_replace:
                            right_cells.remove(right_cell)

                        for right_cell in new_right_cells:
                            right_cells.add(right_cell)

                    else:
                        symetry_offset = 0

                    memory_items = list(left_cells.union(right_cells))

                    # Randomly shuffle these cells
                    random.shuffle(memory_items)


                    # -- Generate File names --
                    id = cur_total_trial_samples
                    memory_stim_fnames = [
                        f"{trial_type}_{str(id).zfill(6)}_memory{str(idx).zfill(2)}.png" for idx in range(list_length)
                    ]

                    probe_stim_fnames = [
                        f"{trial_type}_{str(id).zfill(6)}_probe{str(idx).zfill(2)}.png" for idx in range(1)
                    ]

                    # Adding the trial_data to trials
                    trials_dict[trial_type].append({
                        "trial_type": trial_type,
                        "trial_id": id, 
                        
                        "grid_size": self.grid_size,

                        "list_length": list_length,

                        "gt": gt.value,
                        "symetry_offset": symetry_offset,
                        "memory_items": memory_items,

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

        memory_stims = []

        active_cells = trial["memory_items"]

        for active_cell in active_cells:

            stim, draw = self._init_stim(self.img_config.PRESENTATION_OUTLINE_COLOR)
            self._draw_grid(draw, active_cell)

            memory_stims.append(stim)

        return memory_stims


    def _draw_probe_stims(self, trial):

        probe_items = trial["probe_stim_fnames"]

        probe_stims = []

        for probe_item in probe_items:
            stim, draw = self._init_stim(self.img_config.PROBE_OUTLINE_COLOR)

            probe_stims.append(stim)

        return probe_stims
    
    def _draw_grid(self, draw, active_cell):

        grid_padding = self.config.GRID_PADDING
        grid_cell_size_px = (self.img_size[0] - 2*grid_padding) // self.grid_size


        for cell_idx in range(self.grid_size ** 2):
            
            row = cell_idx // self.grid_size
            col = cell_idx % self.grid_size

            x1 = grid_padding + col * grid_cell_size_px
            y1 = grid_padding + row * grid_cell_size_px

            x2 = x1 + grid_cell_size_px
            y2 = y1 + grid_cell_size_px 

            # Decide the fill color 
            fill_color = self.config.ACTIVE_CELL_COLOR if (row,col) == active_cell else self.config.INACTIVE_CELL_COLOR

            draw.rectangle(
                [x1,y1,x2,y2],
                fill= fill_color,
                outline = self.config.GRID_OUTLINE_COLOR
            )

        return grid_cell_size_px










                        