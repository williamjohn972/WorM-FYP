import os
import json
from PIL import Image, ImageDraw
from enum import Enum

import random, numpy as np, torch

class Colors(Enum):
    RED = "#E80909"
    GREEN = "#10C10D"
    GRAY = "#C2B1B1"
    BLUE = "#3E8AE6"
    ORANGE = "#FA8909"
    PINK = "#FF58BC"
    TEAL = "#04D6E1"
    BLACK = "#000000"
    WHITE = "#FFFFFF"


class ImgConfig():
    IMG_OUTLINE_WIDTH = 3

    BACKGROUND_COLOR = Colors.GRAY.value
    PRESENTATION_OUTLINE_COLOR = Colors.BLUE.value
    PROBE_OUTLINE_COLOR =  Colors.RED.value
    RETENTION_OUTLINE_COLOR = Colors.ORANGE.value
    DISTRACTOR_OUTLINE_COLOR = Colors.GREEN.value
    MEMORY_UPDATE_OUTLINE_COLOR = Colors.PINK.value
    TASK_SWITCH_OUTLINE_COLOR = Colors.TEAL.value
    
class Generator:
    
    trial_types = ["train", "test", "gen_test"]

    img_config = ImgConfig


    def _create_directory(self,path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    
    def _generate_trial_directories(self, data_dir:str, task_type:str):

        self.base_dir = os.path.join(data_dir,task_type)
        return [os.path.join(self.base_dir, trial) for trial in self.trial_types]
        
    def _init_dirs(self, data_dir:str, task_type:str):

        trial_dirs = self._generate_trial_directories(data_dir, task_type)
        for trial_dir in trial_dirs:
            self._create_directory(trial_dir)

        return trial_dirs

    def _save_trial_json(self, trials, fname="trials_data.json"):
        with open(os.path.join(self.base_dir,fname),'w') as f:
            json.dump(trials,f)
    

    def _init_stim(self, phase_color):

        img_size = getattr(self, "img_size")

        memory_stim = Image.new('RGB', img_size, color=self.img_config.BACKGROUND_COLOR)
        draw = ImageDraw.Draw(memory_stim)

        # Draw colored border around image 
        draw.rectangle([0,0, img_size[0] - 1, img_size[1] - 1],
            outline = phase_color,
            width = self.img_config.IMG_OUTLINE_WIDTH)
        
        return memory_stim, draw
    
    
    def _save_stim(self, trial_type, stims, fnames):
        save_file_path = getattr(self, f"{trial_type}_dir")
        for stim, fname in zip(stims, fnames):
            stim.save(os.path.join(save_file_path, fname))


    def _log(self, msg, variant=None):

        if variant: 
            variant = f"- {variant}"

        print(f"[{self.__class__.__name__}]{variant} - {msg}")


    def _log_summary(self, trials_dict, variant = None, extras=None):
        
        if variant: 
            variant = f" ({variant.name.capitalize()})"

        print(f"[{self.__class__.__name__}{variant}] Generation complete")
        
        for k, v in trials_dict.items():
            print(f"  {k}: {len(v)} samples")
        if extras:
            for k, v in extras.items():
                print(f"  {k}: {v}")

        print(f"---------------------\n")
        
