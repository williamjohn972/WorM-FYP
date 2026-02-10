from src.model import Task_Embedding, Show_Task, Memory_Components, Projections
from src.tasks import Tasks

import os

class PathConfig:
    def __init__(
        self,
        root_folder,
        run_name,
        data_folder="./wm_bench_data",
        log_folder="./log/",
        checkpoint_folder="./checkpoints/",
        output_folder="./output",
    ):
        
        run_folder = os.path.join(root_folder, run_name)

        self.data_folder = os.path.join(root_folder,data_folder)
        self.log_folder = os.path.join(run_folder,log_folder)
        self.checkpoint_folder = os.path.join(run_folder,checkpoint_folder)
        self.output_folder = os.path.join(run_folder,output_folder)

class ExecutionConfig:
    def __init__(
        self,
        num_tasks=13,
        gen_test=False,
    ):
        self.num_tasks = num_tasks
        self.gen_test = gen_test

class ResumptionConfig:
    def __init__(
        self,
        resume=False,            
        resume_epoch=0,
    ):
        self.resume = resume
        self.resume_epoch = resume_epoch

class ModelConfig:
    def __init__(
        self,
        *,
        num_tasks: int,

        img_size=96,
        resize_img_size=32,
        num_input_channels=3,
        max_seq_len=20,

        use_cnn=True,

        mem_architecture=Memory_Components.GRU,
        mem_input_size=512,
        mem_hidden_size=96,
        mem_num_layers=1,
        trf_dim_ff=2048,

        task_embedding_type=Task_Embedding.LEARNABLE,
        show_task_time=Show_Task.ALL,

        projection_type=Projections.LINEAR,
    ):
        # basic image / sequence params
        self.img_size = img_size
        self.resize_img_size = resize_img_size
        self.num_input_channels = num_input_channels
        self.max_seq_len = max_seq_len

        # task-related
        self.num_tasks = num_tasks
        self.task_embedding_type = task_embedding_type
        self.show_task_time = show_task_time

        # CNN
        self.use_cnn = use_cnn
        self.final_cnn_output_channels = 512

        # memory module
        self.mem_architecture = mem_architecture
        self.mem_input_size = mem_input_size
        self.mem_hidden_size = mem_hidden_size
        self.mem_num_layers = mem_num_layers
        self.trf_dim_ff = trf_dim_ff

        # projection
        self.projection_type = projection_type

        if self.show_task_time == Show_Task.ALL:
            self.projection_size = self.mem_input_size - self.num_tasks
        else:
            self.projection_size = self.mem_input_size

class TrainConfig:
    def __init__(
        self,
        lr=1e-4,
        batch_size=10,
        num_epochs=200,
        samples_per_task=1200,
        test_interval=5,
    ):
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.samples_per_task = samples_per_task
        self.test_interval = test_interval

class OptimizationConfig:
    def __init__(
        self,
        gpu=0,
        seed=86,
        num_workers=0,
        use_extracted_feats=False,
    ):
        self.gpu = gpu
        self.seed = seed
        self.num_workers = num_workers
        self.use_extracted_feats = use_extracted_feats

class TaskConfig:
    def __init__(self, task_list=None):
        if task_list is None:
            task_list = [
                Tasks.SPATIAL_COORDINATION,
                Tasks.SPATIAL_FREE_RECALL,
                Tasks.SPATIAL_INTEGRATION, #
                Tasks.SPATIAL_MEMORY_UPDATING, #
                Tasks.SPATIAL_TASK_SWITCHING, #

                Tasks.VISUAL_ITEM_RECOGNITION, #
                Tasks.VISUAL_SERIAL_RECALL, #
                Tasks.VISUAL_SERIAL_RECOGNITION,

                Tasks.CHANGE_DETECTION_COLOR,
                Tasks.CHANGE_DETECTION_ORIENTATION,
                Tasks.CHANGE_DETECTION_SIZE,
                Tasks.CHANGE_DETECTION_GAP,
                Tasks.CHANGE_DETECTION_CONJ, #
            ]
        self.task_list = task_list

        self.task_id_map = {task: index for index,task in enumerate(self.task_list)}

class Config:
    def __init__(
        self,
        *,
        root_folder,
        run_name,
        mem_architecture=Memory_Components.GRU,
        task_list=None,
        gpu=0,
        seed=86,
    ):
        # tasks first → defines num_tasks
        self.task_config = TaskConfig(task_list=task_list)
        num_tasks = len(self.task_config.task_list)

        # core configs
        self.path_config = PathConfig(root_folder=root_folder, run_name=run_name)
        self.execution_config = ExecutionConfig(num_tasks=num_tasks)
        self.resumption_config = ResumptionConfig()
        self.train_config = TrainConfig()
        self.optimization_config = OptimizationConfig(gpu=gpu, seed=seed)

        # model config depends on num_tasks
        self.model_config = ModelConfig(
            num_tasks=num_tasks,
            mem_architecture=mem_architecture,
        )


