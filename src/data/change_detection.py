from src.data.generator import Generator, Colors
from PIL import Image, ImageDraw
from typing import List, Dict
from enum import Enum
import random
from copy import deepcopy
from tqdm.auto import tqdm

from src.config import Tasks

class Orientation(Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"

class Recall_Ground_Truth(Enum):
    NO_CHANGE = 0
    CHANGE = 1

class Variant(Enum):
    COLOR = "color"
    ORIENTATION = "orientation"
    SIZE = "size"
    GAP = "gap"
    CONJUNCTION = "conjunction"

class GapType(Enum):
    CONTINUOUS = "continuous"
    BROKEN = "broken"

class BarSize(Enum):
    SMALL = "small"
    LARGE = "large"


class Config():

    BAR_SIZE_SMALL = 9
    BAR_SIZE_LARGE = 15

    BAR_WIDTH = 3

    SMALL_GAP_FRACTIONAL_RANGE = (1/3,2/3)
    LARGE_GAP_FRACTIONAL_RANGE = (2/5, 3/5)

    BAR_MARGIN = 3

    CONJUNCTION_GT_MAP = {
            Variant.COLOR.value: 0,
            Variant.ORIENTATION.value: 1,
            Variant.SIZE.value: 2,
            Variant.GAP.value: 3,
        }

class Change_Detection_Generator(Generator):

    def __init__(self,
                 img_size:int, 
                 set_size_options:List[int], held_out_set_size_options:List[int],
                 retention_interval_options:List[int], held_out_retention_interval_options:List[int],
                 num_samples:int,
                 variant: Variant,
                 data_dir: str = "",
                 save: bool = True, generate_trials = True
                 ):
        
        self.img_size = (img_size,img_size)
        self.variant = variant

        self.conjunction = True if variant == Variant.CONJUNCTION else False

        self.set_size_options = set_size_options
        self.held_out_set_size_options = held_out_set_size_options

        self.retention_interval_options = retention_interval_options
        self.held_out_retention_interval_options = held_out_retention_interval_options

        self.recall_gt_options = [Recall_Ground_Truth.CHANGE, Recall_Ground_Truth.NO_CHANGE]
        self.conjunction_gt_options = [Variant.COLOR, Variant.ORIENTATION, Variant.SIZE, Variant.GAP]

    
        self.config = Config()

        self.num_samples = num_samples
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
            self.train_dir, self.test_dir, self.gen_test_dir = self._init_dirs(self.data_dir, f"{Tasks.CHANGE_DETECTION.name.lower()} - {self.variant.value.lower()}")

        if generate_trials:
            self.generate_trials()

    def generate_trials(self):

        self._log("Generating trials", variant=self.variant)        
        trials_dict = {"train": [], "test": [], "gen_test": []}

        for trial_type in self.trial_types:
            self._log(f"Split: {trial_type}", variant=self.variant)
            
            # set_size_options and retention_interval_options are dependant on the trial type
            if trial_type in ["train", "test"]:

                set_size_options = self.set_size_options
                retention_interval_options = self.retention_interval_options

            else:
                set_size_options = self.held_out_set_size_options
                retention_interval_options = self.held_out_retention_interval_options

            # Each Combination is (set_size, retention_interval)
            total_trial_samples = getattr(self, f"{trial_type}_num_samples")
            total_trial_samples_per_combination = total_trial_samples // (len(self.set_size_options) * len(self.retention_interval_options))
            
            cur_total_trial_samples = 0
            cur_trial_samples_per_combination = {}

            # Loop over each combination
            for set_size in tqdm(set_size_options, leave=False, desc=f"{trial_type} | set_size"):
                for retention_interval in retention_interval_options:

                    combination = (set_size, retention_interval)
                    cur_trial_samples_per_combination[combination] = 0

                    while(cur_trial_samples_per_combination[combination] < total_trial_samples_per_combination):

                        
                        # Generate Memory Array (color, length, orientation)
                        bar_colors = random.choices([Colors.RED, Colors.GREEN], k=set_size)
                        bar_orientations = random.choices(list(Orientation), k=set_size)
                        bar_sizes = random.choices(list(BarSize), k=set_size)
                        bar_gap_types = random.choices(list(GapType), k=set_size)

                        # Resolve Sizes to pixels
                        bar_lengths_px = [self.config.BAR_SIZE_SMALL 
                                       if size == BarSize.SMALL else self.config.BAR_SIZE_LARGE 
                                       for size in bar_sizes]
                        
                        # Resolve Gaps to pixel ranges 
                        bar_gap_fractional_ranges = [self.config.SMALL_GAP_FRACTIONAL_RANGE
                                                     if size == BarSize.SMALL else self.config.LARGE_GAP_FRACTIONAL_RANGE
                                                     for size in bar_sizes]
                        

                        # For each bar choose a random x,y from (img_size, img_size)

                        memory_items = []
                        for idx in range(set_size):

                            cur_color = bar_colors[idx]
                            cur_orientation = bar_orientations[idx]
                            cur_size = bar_sizes[idx]
                            cur_gap_type = bar_gap_types[idx]
                            cur_length = bar_lengths_px[idx]
                            
                            # Lets store each bar as dict
                            cur_bar_center_coords = self._create_bar(memory_items, cur_orientation.value, cur_length, max_tries=100)
                            cur_bar_coords = self._compute_bar_coords(cur_bar_center_coords, cur_orientation.value, cur_length)

                            if (cur_bar_center_coords) is None:
                                raise RuntimeError("Failed to place bars - restart trial")
                            
                            # Compute Gaps
                            cur_bar_gap_fractional_range = bar_gap_fractional_ranges[idx]
                            if cur_gap_type == GapType.BROKEN:
                                cur_gap_coords = self._compute_gap_coords(cur_bar_coords, cur_orientation.value, cur_bar_gap_fractional_range)     

                            else:
                                cur_gap_coords = (None, None, None, None)

                            # Store Memory Items
                            memory_items.append({
                                    "x_center": cur_bar_center_coords[0],
                                    "y_center": cur_bar_center_coords[1],

                                    "x1": cur_bar_coords[0],
                                    "y1": cur_bar_coords[1],
                                    "x2": cur_bar_coords[2],
                                    "y2": cur_bar_coords[3],
                                    
                                    "gap_type": cur_gap_type.value,
                                    "gap_fractional_range": cur_bar_gap_fractional_range,
                                    "gap_x1": cur_gap_coords[0],
                                    "gap_y1": cur_gap_coords[1],
                                    "gap_x2": cur_gap_coords[2],
                                    "gap_y2": cur_gap_coords[3],

                                    "color": cur_color.value,

                                    "orientation": cur_orientation.value,

                                    "size": cur_size.value,
                                    "length": cur_length,
                                    "width": self.config.BAR_WIDTH,
                                    
                            })
                    

                        # Probe generation logic 
                        probe_items = deepcopy(memory_items)

                        # First we need to choose the ground truth
                        recall_gt = random.choice(list(Recall_Ground_Truth))
                        conjunction_gt = None

                        # If CHANGE, we need to choose one bar
                        if recall_gt == Recall_Ground_Truth.CHANGE:
                            probe_bar_idx = random.choice(range(set_size))
                            probe_bar = probe_items[probe_bar_idx]
                            probe_bar_center_coords = probe_bar["x_center"],probe_bar["y_center"]
                            probe_bar_coords = probe_bar["x1"],probe_bar["y1"],probe_bar["x2"],probe_bar["y2"]
                            
                    
                            if self.conjunction:
                                temp_variant = random.choice([
                                    Variant.COLOR, Variant.GAP, Variant.SIZE, Variant.ORIENTATION
                                ])
                            else:
                                temp_variant = self.variant
                            
                            if temp_variant == Variant.COLOR:

                                if probe_bar["color"] == Colors.RED.value:
                                    probe_bar["color"] = Colors.GREEN.value
                                else:
                                    probe_bar["color"] = Colors.RED.value
                            

                            elif temp_variant == Variant.GAP:
                                
                                if probe_bar["gap_type"] == GapType.BROKEN.value:
                                    probe_bar["gap_type"] = GapType.CONTINUOUS.value
                                    probe_bar["gap_x1"], probe_bar["gap_y1"], probe_bar["gap_x2"], probe_bar["gap_y2"] = (None,None,None,None)
                                    
                                else:
                                    probe_bar["gap_type"] = GapType.BROKEN.value
                                    probe_bar["gap_x1"], probe_bar["gap_y1"], probe_bar["gap_x2"], probe_bar["gap_y2"] = self._compute_gap_coords(probe_bar_coords, probe_bar["orientation"], probe_bar["gap_fractional_range"])
                                    
                                    

                                
                            elif temp_variant == Variant.ORIENTATION:

                                if probe_bar["orientation"] == Orientation.HORIZONTAL.value:
                                    probe_bar["orientation"] = Orientation.VERTICAL.value
                                else:
                                    probe_bar["orientation"] = Orientation.HORIZONTAL.value

                                # Recompute coords
                                probe_bar_coords = self._compute_bar_coords(probe_bar_center_coords, probe_bar["orientation"], probe_bar["length"])
                                probe_bar["x1"],probe_bar["y1"],probe_bar["x2"],probe_bar["y2"] = probe_bar_coords

                                # Recompute Gap Coords
                                probe_bar["gap_x1"], probe_bar["gap_y1"], probe_bar["gap_x2"], probe_bar["gap_y2"] = self._compute_gap_coords(probe_bar_coords, probe_bar["orientation"], probe_bar["gap_fractional_range"])
                                
                            elif temp_variant == Variant.SIZE:

                                if probe_bar["size"] == BarSize.LARGE.value:
                                    probe_bar["size"] = BarSize.SMALL.value
                                    probe_bar["length"] = self.config.BAR_SIZE_SMALL
                                    probe_bar["gap_fractional_range"] = self.config.SMALL_GAP_FRACTIONAL_RANGE
                                
                                else:
                                    probe_bar["size"] = BarSize.LARGE.value
                                    probe_bar["length"] = self.config.BAR_SIZE_LARGE
                                    probe_bar["gap_fractional_range"] = self.config.LARGE_GAP_FRACTIONAL_RANGE


                                # Recompute coords
                                probe_bar_coords = self._compute_bar_coords(probe_bar_center_coords, probe_bar["orientation"], probe_bar["length"])
                                probe_bar["x1"],probe_bar["y1"],probe_bar["x2"],probe_bar["y2"] = probe_bar_coords

                                # Recompute Gap Coords
                                probe_bar["gap_x1"], probe_bar["gap_y1"], probe_bar["gap_x2"], probe_bar["gap_y2"] = self._compute_gap_coords(probe_bar_coords, probe_bar["orientation"], probe_bar["gap_fractional_range"])

                            if self.conjunction:
                                conjunction_gt = temp_variant.value


                        # -- Generate File Names --
                        id = cur_total_trial_samples
                        memory_stim_fnames = [
                            f"{trial_type}_{str(id).zfill(6)}_memory{str(idx).zfill(2)}.png" for idx in range(1)
                        ]

                        retention_stim_fnames = [
                            f"{trial_type}_{str(id).zfill(6)}_retention{str(idx).zfill(2)}.png" for idx in range(retention_interval)
                        ]

                        probe_stim_fnames = [
                            f"{trial_type}_{str(id).zfill(6)}_probe{str(idx).zfill(2)}.png" for idx in range(1)
                        ]

                        # Adding the trial_data to trials
                        trials_dict[trial_type].append({
                            "trial_type": trial_type,
                            "trial_id": id, 
                            
                            "set_size": set_size,
                            "retention_interval": retention_interval,

                            "memory_items": memory_items,

                            "probe_items": probe_items,
                             
                            "recall_gt": recall_gt.value,
                            "conjunction_gt": conjunction_gt,

                            "memory_stim_fnames": memory_stim_fnames,
                            "retention_stim_fnames": retention_stim_fnames,
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


    def _compute_gap_coords(self,bar_coords, orientation, gap_fractional_range):

        start_frac, end_frac = gap_fractional_range
        x1,y1,x2,y2 = bar_coords

        if orientation == Orientation.HORIZONTAL.value:
            bar_length = x2 - x1

            gap_start = x1 + start_frac * bar_length
            gap_end = x1 + end_frac * bar_length

            return (gap_start, y1, gap_end, y2)

        else:
            bar_length = y2 - y1

            gap_start = y1 + start_frac * bar_length
            gap_end = y1 + end_frac * bar_length

            return (x1, gap_start, x2, gap_end)

    def _create_bar(self, 
                   bar_center_coords,
                   orientation, length,
                   max_tries=100):
        
        try_count = 0

        while try_count < max_tries:
        
            x = random.choice(range(self.img_size[0]))
            y = random.choice(range(self.img_size[1]))

            # Calculate the ends of the bar based on orientation
            cur_bar_coords = self._compute_bar_coords((x,y), orientation, length)

            # Check if the line is within the img
            if self._is_valid_coords(cur_bar_coords, bar_center_coords):
                return (x,y)
            
            try_count += 1


    def _compute_bar_coords(self, bar_center_coords, orientation, length):
        x, y = bar_center_coords
        half_len = length / 2

        if orientation == Orientation.HORIZONTAL.value:
            x1 = x - half_len
            y1 = y
            x2 = x + half_len
            y2 = y

        else: 
            x1 = x
            y1 = y - half_len
            x2 = x
            y2 = y + half_len

        return (x1, y1, x2, y2)

                 
    def _is_valid_coords(self, bar_coords, bar_list):

        x1,y1,x2,y2 = bar_coords

        valid = (x1 >= self.config.BAR_MARGIN 
                 and x2 <= self.img_size[0] - self.config.BAR_MARGIN
                 and y1 >= self.config.BAR_MARGIN 
                 and y2 <= self.img_size[1] - self.config.BAR_MARGIN)
        
        if valid and len(bar_list) == 0:
            return True
        
        if valid and not self._has_overlap(bar_coords, bar_list, self.config.BAR_MARGIN):
            return True
        
        return False

    def _has_overlap(self, bar1, bar_list, padding = 0):

        x1a, y1a, x2a, y2a = bar1
        x1a -= padding; y1a -= padding
        x2a += padding; y2a += padding

        for bar in bar_list:

            x1b, y1b, x2b, y2b = bar["x1"], bar["y1"], bar["x2"], bar["y2"]

            x1b -= padding; y1b -= padding
            x2b += padding; y2b += padding

            x_overlap = (x1a < x2b) and (x1b < x2a)
            y_overlap = (y1a < y2b) and (y1b < y2a)

            if x_overlap and y_overlap:
                return True 

        return False

    def _draw_trial_stims(self, trials):

        trials_dict = {"train": [], "test": [], "gen_test": []}

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

        memory_stims = []

        stim, draw = self._init_stim(self.img_config.PRESENTATION_OUTLINE_COLOR)

        bars = trial["memory_items"]

        for bar in bars:

            x1,y1,x2,y2 = bar["x1"],bar["y1"],bar["x2"],bar["y2"]
            bar_color = bar["color"]
            bar_width = bar["width"]

            gap_type = bar["gap_type"]
            gap_x1 = bar["gap_x1"]
            gap_y1 = bar["gap_y1"]
            gap_x2 = bar["gap_x2"]
            gap_y2 = bar["gap_y2"]

            draw.line(
                (x1, y1, x2, y2),
                fill=bar_color,
                width=bar_width
            )

            if gap_type == GapType.BROKEN.value:

                draw.line(
                    (gap_x1, gap_y1, gap_x2, gap_y2),
                    fill= self.img_config.BACKGROUND_COLOR,
                    width=bar_width
                )
        

        memory_stims.append(stim)

        return memory_stims

    def _draw_retention_stims(self, trial):

        retention_interval = trial["retention_interval"]

        stims = []

        for _ in range(retention_interval):
            stim, _ = self._init_stim(self.img_config.RETENTION_OUTLINE_COLOR)
            stims.append(stim)

        return stims
    

    def _draw_probe_stims(self, trial):

        probe_stims = []

        stim, draw = self._init_stim(self.img_config.PROBE_OUTLINE_COLOR)

        bars = trial["probe_items"]

        for bar in bars:

            x1,y1,x2,y2 = bar["x1"],bar["y1"],bar["x2"],bar["y2"]
            bar_color = bar["color"]
            bar_width = bar["width"]

            gap_type = bar["gap_type"]
            gap_x1 = bar["gap_x1"] 
            gap_y1 = bar["gap_y1"]
            gap_x2 = bar["gap_x2"]
            gap_y2 = bar["gap_y2"]

            draw.line(
                (x1, y1, x2, y2),
                fill=bar_color,
                width=bar_width
            )

            if gap_type == GapType.BROKEN.value:

                draw.line(
                    (gap_x1, gap_y1, gap_x2, gap_y2),
                    fill= self.img_config.BACKGROUND_COLOR,
                    width=bar_width
                )
        

        probe_stims.append(stim)

        return probe_stims