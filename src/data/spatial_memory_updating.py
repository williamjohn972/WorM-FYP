from src.data.generator import Generator, Colors
from enum import Enum
from typing import List
import random
import math
from copy import deepcopy

from src.tasks import Tasks



class Direction(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"

    UP_LEFT = "up-left"
    UP_RIGHT = "up-right"
    DOWN_LEFT = "down-left"
    DOWN_RIGHT = "down-right"


class Config():

    BOX_DRAW_RADIUS = 35
    BOX_SIZE = 25

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

    BOX_COLOR = Colors.GREEN.value
    PROBE_BOX_COLOR = Colors.RED.value
    MARKER_COLOR = Colors.RED.value
    ARROW_COLOR = Colors.BLACK.value

    MARKER_RADIUS = 2
    ARROW_WIDTH = 1

class Spatial_Memory_Updating_Generator(Generator):

    def __init__(self,
                 grid_size: int, img_size: int,
                 set_size_options: List[int], held_out_set_size_options: List[int],
                 num_updates_options: List[int], presentation_time_options: List[int], 
                 num_samples: int, 
                 data_dir: str = "", save: bool = True, generate_trials=True
                 ):
        
        self.config = Config()
        
        self.grid_size = grid_size
        self.img_size = (img_size,img_size)

        self.set_size_options = set_size_options
        self.held_out_set_size_options = held_out_set_size_options

        self.num_updates_options = num_updates_options
        self.presentation_time_options = presentation_time_options

        self.train_num_samples = num_samples 
        self.test_num_samples = (int) (num_samples * 0.1)
        self.gen_test_num_samples = (int) (num_samples * 0.1)

        self.data_dir = data_dir
        self.save = save

        # Assertions
        assert self.train_num_samples % len(set_size_options) == 0
        assert self.test_num_samples % len(set_size_options) == 0
        assert self.gen_test_num_samples % len(held_out_set_size_options) == 0
            
        if self.save:
            self.train_dir, self.test_dir, self.gen_test_dir = self._init_dirs(self.data_dir, Tasks.SPATIAL_MEMORY_UPDATING.name.lower())

        if generate_trials:
            self.generate_trials()
    
    def generate_trials(self):
        
        trials_dict = {"train": [], "test": [], "gen_test": []}

        for trial_type in self.trial_types:
            
            # set_size_options is dependant on the trial type
            if trial_type in ["train", "test"]:

                set_size_options = self.set_size_options

            else:
                set_size_options = self.held_out_set_size_options

            # Each Combination is (set_size, presentation_time)
            total_trial_samples = getattr(self, f"{trial_type}_num_samples")
            total_trial_samples_per_combination = total_trial_samples // (len(set_size_options) * len(self.presentation_time_options))
            
            cur_total_trial_samples = 0
            cur_trial_samples_per_combination = {}

            # Loop over each combination
            for set_size in set_size_options:
                for presentation_time in self.presentation_time_options:

                    combination = (set_size, presentation_time)
                    cur_trial_samples_per_combination[combination] = 0

                    while(cur_trial_samples_per_combination[combination] < total_trial_samples_per_combination):

                        # Randomly Choose the Number of Memory Updates
                        num_updates = random.choice(self.num_updates_options)

                        # Generate Box Locations
                        box_center_coords, box_grid_center_coords = self._make_box_grids(set_size)

                        # Initialise Marker positions within each box 
                        init_marker_state = {}
                        for box_idx, _ in box_center_coords.items():

                            # Choose a grid cell
                            cell_idx = random.choice(range(self.grid_size ** 2))
                            init_marker_state[box_idx] = cell_idx

                        # Generate Updates 
                        updates = []
                        marker_states = [deepcopy(init_marker_state)]

                        for count in range(num_updates):
                            
                            # Cycle through the boxes 
                            chosen_box = count % set_size

                            # Current marker state
                            cur_marker_state = marker_states[-1]

                            # Choose a random direction and validate the maker update
                            update_dir, new_marker_grid_cell = self._update_marker_state(chosen_box, cur_marker_state)

                            updates.append((chosen_box, update_dir.value))

                            next_marker_state = deepcopy(cur_marker_state)
                            next_marker_state[chosen_box] = new_marker_grid_cell
                            marker_states.append(next_marker_state)


                        # --- Probe Phase --
                        
                        # Choose the box order for testing
                        probe_order = random.sample(range(set_size), set_size)
                        
                        # Generate the ground truth
                        final_marker_state = marker_states[-1]
                        probe_gt_order = [final_marker_state[probe_box_idx] for probe_box_idx in probe_order]


                        # -- Generate File names --
                        id = cur_total_trial_samples
                        memory_stim_fnames = [
                            f"{trial_type}_{str(id).zfill(6)}_memory{str(idx).zfill(2)}.png" for idx in range(1)
                        ]

                        update_stim_fnames = [
                            f"{trial_type}_{str(id).zfill(6)}_update_{str(idx).zfill(2)}.png" for idx in range(num_updates)
                        ]

                        probe_stim_fnames = [
                            f"{trial_type}_{str(id).zfill(6)}_probe{str(idx).zfill(2)}.png" for idx in range(set_size)
                        ]

                        # Adding the trial_data to trials
                        trials_dict[trial_type].append({
                            "trial_type": trial_type,
                            "trial_id": id, 
                            
                            "grid_size": self.grid_size,

                            "set_size": set_size,
                            "presentation_time": presentation_time,

                            "num_updates": num_updates,

                            "box_center_coords": box_center_coords,
                            "box_grid_center_coords": box_grid_center_coords,

                            "initial_marker_state": init_marker_state,
                            "marker_states": marker_states,
                            "updates": updates,
                            "final_marker_state": final_marker_state,
                            

                            "probe_order": probe_order,
                            "probe_gt_order": probe_gt_order,

                            "memory_stim_fnames": memory_stim_fnames,
                            "update_stim_fnames": update_stim_fnames,
                            "probe_stim_fnames": probe_stim_fnames,
                        })

                        # Update Counters
                        cur_total_trial_samples += 1
                        cur_trial_samples_per_combination[combination] += 1

        if self.save:
            self._save_trial_json(trials_dict)

        self._draw_trial_stims(trials_dict)

        return trials_dict



    def _make_box_grids(self, set_size):
        
        box_center_coords = {}
        box_grid_center_coords = {}

        img_width = self.img_size[0]
        img_height = self.img_size[1]

        x_center = img_width / 2
        y_center = img_height / 2

        # If the set_size is 1 we simply place the box at the center 
        if set_size == 1:
            box_center_coords[0] = (x_center,y_center)

        # Else we want to arange each box around the center of the image
        # in a circular pattern
        elif set_size > 1:

            for box_idx in range(set_size):

                angle = (2* math.pi * box_idx) / set_size
                x = x_center + (self.config.BOX_DRAW_RADIUS * math.cos(angle))
                y = y_center + (self.config.BOX_DRAW_RADIUS * math.sin(angle))

                box_center_coords[box_idx] = (x,y)
                

        # Now we want to csreate a grid inside each box
        for box_idx, (x,y) in box_center_coords.items():
            
            box_grid_center_coords[box_idx] = {}

            cell_size = self.config.BOX_SIZE / self.grid_size

            start_x = x - 0.5 * cell_size * (self.grid_size - 1)
            start_y = y - 0.5 * cell_size * (self.grid_size - 1)

            for row in range(self.grid_size):
                for col in range(self.grid_size):

                    cell_idx = row * self.grid_size + col
                    cell_center_x = start_x + col * cell_size
                    cell_center_y = start_y + row * cell_size

                    box_grid_center_coords[box_idx][cell_idx] = (cell_center_x,cell_center_y)

        return box_center_coords, box_grid_center_coords
        
    def _update_marker_state(self,chosen_box_idx, init_marker_state, max_tries = 100):
        
        count = 0

        while count < max_tries:

            grid_cell = init_marker_state[chosen_box_idx]

            row = grid_cell // self.grid_size
            col = grid_cell % self.grid_size
            
            update_dir = random.choice(list(Direction))
            row_diff, col_diff = self.config.DIRECTION_MATH[update_dir]
            
            new_row = row + row_diff
            new_col = col + col_diff

            if (0 <= new_row < self.grid_size
                and 0 <= new_col < self.grid_size):
                
                new_grid_cell = new_row * self.grid_size + new_col
                return update_dir, new_grid_cell

            else:
                count += 1

        raise RuntimeError("No valid update direction is found")


    def _draw_trial_stims(self, trials):

        trials_dict = {"train": [], "test": [], "gen_test": []}

        for trial_type in self.trial_types:

            trials_list = trials[trial_type]

            # We go over each trial 
            for trial in trials_list:

                # Draw the memory stim 
                memory_stims = self._draw_memory_stims(trial)

                # Draw the retention stim
                update_stims = self._draw_update_stims(trial)

                # Draw the probe stim 
                probe_stims = self._draw_probe_stims(trial)

                # Saving Logic 
                if self.save:
                    self._save_stim(trial_type, memory_stims, trial["memory_stim_fnames"])
                    self._save_stim(trial_type, update_stims, trial["update_stim_fnames"])
                    self._save_stim(trial_type, probe_stims, trial["probe_stim_fnames"])


    def _draw_memory_stims(self, trial):

        box_center_coords = trial["box_center_coords"]
        box_grid_center_coords = trial["box_grid_center_coords"]

        memory_stims = []

        init_marker_state = trial["initial_marker_state"]

        stim, draw = self._init_stim(self.img_config.PRESENTATION_OUTLINE_COLOR)
        self._draw_boxes(draw, box_center_coords, None)
        self._draw_markers(draw, box_grid_center_coords, init_marker_state)

        
        memory_stims.append(stim)

        return memory_stims

    def _draw_update_stims(self, trial):

        box_center_coords = trial["box_center_coords"]
        box_grid_center_coords = trial["box_grid_center_coords"]

        num_updates = trial["num_updates"]
        marker_states = trial["marker_states"]
        updates = trial["updates"]

        stims = []

        for idx in range(num_updates):
            stim, draw = self._init_stim(self.img_config.MEMORY_UPDATE_OUTLINE_COLOR)
            self._draw_boxes(draw, box_center_coords, None)

            update = updates[idx]
            prev_marker_coords = marker_states[idx]
            next_marker_coords = marker_states[idx+1]
            self._draw_update_dir(draw, update, box_grid_center_coords, prev_marker_coords, next_marker_coords)

            stims.append(stim)

        return stims
    

    def _draw_probe_stims(self, trial):

        box_center_coords = trial["box_center_coords"]
        box_grid_center_coords = trial["box_grid_center_coords"]

        probe_order = trial["probe_order"]

        probe_stims = []

        for probe_box_idx,_ in enumerate(probe_order):
            stim, draw = self._init_stim(self.img_config.PROBE_OUTLINE_COLOR)

            self._draw_boxes(draw, box_center_coords, probe_box_idx)

            probe_stims.append(stim)


        return probe_stims
    
    def _draw_boxes(self, draw, box_center_coords, probe_box_idx = None):
        
        half = self.config.BOX_SIZE // 2

        for box_idx, (x,y) in box_center_coords.items():
            
            if probe_box_idx == None or box_idx != probe_box_idx:
                fill_color = self.config.BOX_COLOR

            else:
                fill_color = self.config.PROBE_BOX_COLOR

            draw.rectangle(
                [x - half, y - half, x + half, y + half],
                fill = fill_color
            )

    def _draw_markers(self, draw, box_grid_center_coords, marker_coords):
        
        for box, grid_cell in marker_coords.items():

            r = self.config.MARKER_RADIUS

            x,y = box_grid_center_coords[box][grid_cell]

            draw.ellipse([x-r, y-r, x+r, y+r],
                         fill = self.config.MARKER_COLOR)


    def _draw_update_dir(self, draw, update, box_grid_center_coords, prev_marker_coords, next_marker_coords):
        
            box, dir = update
            
            prev_grid_cell = prev_marker_coords[box]
            prev_x, prev_y = box_grid_center_coords[box][prev_grid_cell]

            next_grid_cell = next_marker_coords[box]
            next_x, next_y = box_grid_center_coords[box][next_grid_cell]

            draw.line([prev_x, prev_y, next_x, next_y],
                      fill=self.config.ARROW_COLOR,
                      width=self.config.ARROW_WIDTH)


            # arrowhead 
            angle = math.atan2(next_y - prev_y, next_x - prev_x)
            head_len = 4
            head_angle = math.pi / 6

            left = (
                next_x - head_len * math.cos(angle - head_angle),
                next_y - head_len * math.sin(angle - head_angle),
            )
            right = (
                next_x - head_len * math.cos(angle + head_angle),
                next_y - head_len * math.sin(angle + head_angle),
            )

            draw.line([next_x, next_y, *left], fill=self.config.ARROW_COLOR, width=self.config.ARROW_WIDTH)
            draw.line([next_x, next_y, *right], fill=self.config.ARROW_COLOR, width=self.config.ARROW_WIDTH)
        

        
    



                        




