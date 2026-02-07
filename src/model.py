import torch 
import torch.nn as nn

from src.tasks import Tasks, task_id_map

from enum import Enum

class Projections(Enum):
    LINEAR = "linear"
    NON_LINEAR = "non_linear"

class Task_Embedding(Enum):
    STATIC = "static"
    LEARNABLE = "learnable"

class Show_Task(Enum):

    ALL = "all"
    START = "start"

class Memory_Components(Enum):
    RNN = "RNN"
    GRU = "GRU"
    LSTM = "LSTM"
    TRANSFORMER = "TRF"


class WM_Model(nn.Module):

    def __init__(self, config, device: str):

        super().__init__()

        self.config = config
        self.device = device

        # Create CNN Encoder and Projection Head
        if self.config.use_cnn:

            self.CNN_Encoder = self._init_cnn_encoder(num_input_channels= self.config.num_input_channels,
                                               num_output_channels = self.config.final_cnn_output_channels)

            self.Projection_Head = self._init_projection_head(projection_type = self.config.projection_type, 
                                                       in_features = self.config.final_cnn_output_channels,
                                                       out_features = self.config.projection_size)

        # Create Task Embeddings 
        if self.config.show_task_time == Show_Task.START:
            self.Task_Embeddings = self._init_task_embeddings(num_embeddings = self.config.num_tasks,
                                                        embedding_dim = self.config.mem_input_size,
                                                        task_embedding_type = self.config.task_embedding_type,
                                                        num_tasks = self.config.num_tasks)
            
        elif self.config.show_task_time  == Show_Task.ALL:
            self.Task_Embeddings = self._init_task_embeddings(num_embeddings = self.config.num_tasks,
                                                    embedding_dim = self.config.num_tasks,
                                                    task_embedding_type = self.config.task_embedding_type,
                                                    num_tasks = self.config.num_tasks)
        
        
        
        
        # Build the Memory Component and its corresponding Classifier Head
        if self.config.mem_architecture in [Memory_Components.RNN, Memory_Components.GRU, Memory_Components.LSTM]:
            self.Memory = self._init_rnn_memory_component(type = self.config.mem_architecture,
                                                      input_size = self.config.mem_input_size,
                                                      hidden_size = self.config.mem_hidden_size,
                                                      num_layers = self.config.mem_num_layers,
                                                      batch_first = True)
            
            # Initialise Weights 
            self._init_weights()

            # Create Classifier Head Map
            self.Classifier_Head_Map = self._init_classifier_heads(in_features = self.config.mem_hidden_size)  


        elif self.config.mem_architecture == Memory_Components.TRANSFORMER:
            self.Positional_Embeddings, self.Memory = self._init_trf_memory_component(num_embeddings = self.config.max_seq_len,
                                                                                  embedding_dim = self.config.mem_input_size,
                                                                                  d_model = self.config.mem_input_size,
                                                                                  n_head = 8,
                                                                                  num_layers = self.config.mem_num_layers,
                                                                                  dim_feedforward = self.config.trf_dim_ff,
                                                                                  batch_first = True)

            # Create Classifier Head Map 
            self.Classifier_Head_Map = self._init_classifier_heads(in_features = self.config.mem_input_size)


    def forward(self, X:torch.Tensor, task: Tasks, actual_sequence_length = None):
        
        # We need to reshape accordingly to CNN from (B,T,C,H,W) --> (B*T,C,H,W)
        if self.config.use_cnn:

            batch_size, sequence_length, num_channels, height, width = X.size()

            X = X.reshape((batch_size * sequence_length, num_channels, height, width))

            cnn_output = self.CNN_Encoder(X) # Shape --> (batch_size * seq_length, 512, 1, 1)
            cnn_output = cnn_output.reshape(batch_size * sequence_length, self.config.final_cnn_output_channels)

            # Now we need to pass the cnn output through the projection head
            projection_output = self.Projection_Head(cnn_output)
            # reshape it from (B*T, projection_size) --> (B, T, projection_size)
            projection_output = projection_output.reshape(batch_size, sequence_length, self.config.projection_size)
        
        else:
            cnn_output = None
            batch_size = X.shape[0]
            sequence_length = X.shape[1]

            projection_output = X.reshape(batch_size, sequence_length, self.config.projection_size)
    
        # First we need to convert our task to a task id 
        task_id = task_id_map[task]
        task_idx = torch.tensor([task_id], dtype = torch.long).to(self.device)
        # Lookup Task Embedding
        task_embedding = self.Task_Embeddings(task_idx)
        
        # Current task embedding shape --> (1, embed_dim)
        # There are two ways we add the embedding to the projection. 
        # It depends whether we want to show it for every image in the sequence
        # or only at the beginning of the sequence
        if self.config.show_task_time == Show_Task.ALL:
            # new shape --> (B,T, mem_input_size)
            # mem_input_size = projection_size + task_embedding_size
            task_embedding = task_embedding.repeat(batch_size, sequence_length, 1)
            projection_output = torch.cat((task_embedding, projection_output), dim = 2)

        elif self.config.show_task_time == Show_Task.START:
            # new shape --> (B, T + 1, projection_size)
            task_embedding = task_embedding.repeat(batch_size, 1, 1)
            projection_output = torch.cat((task_embedding, projection_output), dim = 1)

            # update sequence length
            sequence_length = projection_output.shape[1]


        mem_input = projection_output
        # Now its time to pass everything into the memory component 
        # for RNN, GRU, LSTM we can directly pass this into their architecture
        if self.config.mem_architecture in [Memory_Components.GRU, Memory_Components.RNN]:
            self._flatten_mem_params()
            mem_output, mem_h_n = self.Memory(mem_input)
            mem_output_reshaped = mem_output.reshape(-1, self.config.mem_hidden_size)


        elif self.config.mem_architecture == Memory_Components.LSTM:
            self._flatten_mem_params()
            mem_output, (mem_h_n, mem_c_n) = self.Memory(mem_input)
            mem_output_reshaped = mem_output.reshape(-1, self.config.mem_hidden_size)


        # if the memory is a TRF then we need to account for the positional embeddings
        elif self.config.mem_architecture == Memory_Components.TRANSFORMER:
            # The transformer needs two things to behave like memory (Positional Embeddings and Attention Mask)
            # First we add the Positional Embeddings, shape --> (max_seq_len, mem_input_size)
            positions = torch.arange(sequence_length).unsqueeze(0).repeat(batch_size, 1).to(self.device)
            # positional_embeddings, shape --> (T, mem_input_size)
            positional_embeddings = self.Positional_Embeddings(positions)

            # now we add the positional embeddings to the mem_input
            mem_input = mem_input + positional_embeddings

            # now we need to create the mask so that the Transformer cannot see timestep t+1 before t
            mask = nn.Transformer.generate_square_subsequent_mask(sz=sequence_length).to(self.device)

            if actual_sequence_length != None:
                src_key_padding_mask = torch.zeros((batch_size, sequence_length)).bool().to(self.device)
                for i in range(batch_size):
                    src_key_padding_mask[i, actual_sequence_length[i]:] = True

            else: 
                src_key_padding_mask = None

            mem_output = self.Memory(mem_input, mask = mask, src_key_padding_mask = src_key_padding_mask)
            mem_output_reshaped = mem_output.reshape(-1, self.config.mem_input_size)
            mem_h_n = None

        else: 
            raise ValueError(f"Invalid Memory Component Type {self.config.mem_architecture.value}")
        

        # Finally we want to pass the mem_output into the appropriate classifier head
        Classifier = self.Classifier_Head_Map[task.name]
        output = Classifier(mem_output_reshaped)
        output = output.reshape(batch_size, sequence_length, -1)

        
        return output, mem_output, mem_h_n, projection_output, cnn_output
    
            
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def _init_cnn_encoder(self, num_input_channels, num_output_channels):

        # First We need to make the CNN Encoder
        # Input (B,T,C,H,W) ---> we need to make it in forward(B*T,C,H,W)
        # Outut (B*T, 512, 1,1) ---> we need to make it forward(B*T, 512) 
        CNN_Encoder = nn.Sequential(
            nn.Conv2d(in_channels= num_input_channels, out_channels= 64, 
                        kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),

            nn.Conv2d(in_channels = 64, out_channels = 128, 
                        kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(in_channels = 128, out_channels = 256, 
                        kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(in_channels = 256, out_channels = num_output_channels, 
                        kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size = (1,1)),
        )

        return CNN_Encoder

    def _init_projection_head(self, projection_type, in_features, out_features):

        # Next we need the Projection Head
        # This is the bridge between the Perception (CNN) and the Memory (RNN, LSTM, GRU, TRF)
        # mem_input_size = projection_size (img_features) + num_tasks (task_embed dims)
        
        # Projection Head Depends on he Projection Type
        if projection_type == Projections.LINEAR:
            Projection_Head = nn.Linear(in_features = in_features, 
                                        out_features = out_features)

        elif projection_type == Projections.NON_LINEAR:
            Projection_Head = nn.Sequential(
                nn.Linear(in_features = in_features,
                            out_features = 256),
                nn.ReLU(),
                nn.Linear(in_features = 256,
                            out_features = out_features)
            )

        else:
            raise ValueError("Not a Valid Projection Type")

        return Projection_Head

    def _init_task_embeddings(self, num_embeddings, embedding_dim, task_embedding_type, num_tasks):
        # Create Task Embeddings 
        Task_Embeddings = nn.Embedding(num_embeddings = num_embeddings,
                                            embedding_dim = embedding_dim)
        
        # We've got two modes for the embeddings (static and learnable)
        if task_embedding_type == Task_Embedding.STATIC:
            Task_Embeddings.weight.data = torch.eye(num_tasks)
            Task_Embeddings.weight.requires_grad = False

        elif task_embedding_type == Task_Embedding.LEARNABLE:
            Task_Embeddings.weight.requires_grad = True

        else:
            raise ValueError("Invalid Task Embedding Type")
        
        return Task_Embeddings

    def _init_rnn_memory_component(self, type, input_size, hidden_size, num_layers, batch_first):
        # Build the Memory Component

        memory_map = {
            Memory_Components.RNN: nn.RNN,
            Memory_Components.GRU: nn.GRU, 
            Memory_Components.LSTM: nn.LSTM
        }

        rnn_class = memory_map.get(type)

        if rnn_class is None:
            raise ValueError(f"Invalid Memory Type: {type.value}")

        Memory = rnn_class(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first = batch_first)
        
        return Memory

    def _init_trf_memory_component(self, num_embeddings, embedding_dim, d_model, n_head, dim_feedforward, num_layers, batch_first):
        # We need Positional Embeddings for the transformer 
        Positional_Embeddings = nn.Embedding(num_embeddings =  num_embeddings, 
                                            embedding_dim = embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model = d_model, 
                                                nhead = n_head, 
                                                dim_feedforward = dim_feedforward,
                                                batch_first = batch_first)
        
        Memory = nn.TransformerEncoder(encoder_layer = encoder_layer, num_layers = num_layers)

        return Positional_Embeddings, Memory

    def _init_classifier_heads(self, in_features):
        
        Classifier_Head_Map = nn.ModuleDict({
            Tasks.SPATIAL_COORDINATION.name: nn.Linear(in_features, out_features = 1),
            Tasks.SPATIAL_FREE_RECALL.name: nn.Linear(in_features, out_features = 100),
            Tasks.SPATIAL_INTEGRATION.name: nn.Linear(in_features, out_features = 1),
            Tasks.SPATIAL_MEMORY_UPDATING.name: nn.Linear(in_features, out_features = 9),
            Tasks.SPATIAL_TASK_SWITCHING.name: nn.Linear(in_features, out_features = 3), 

            Tasks.VISUAL_ITEM_RECOGNITION.name: nn.Linear(in_features, out_features = 1),
            Tasks.VISUAL_SERIAL_RECALL.name: nn.Linear(in_features, out_features = 9),
            Tasks.VISUAL_SERIAL_RECOGNITION.name: nn.Linear(in_features, out_features = 1),

            Tasks.CHANGE_DETECTION_COLOR.name: nn.Linear(in_features, out_features = 1),
            Tasks.CHANGE_DETECTION_ORIENTATION.name: nn.Linear(in_features, out_features = 1),
            Tasks.CHANGE_DETECTION_GAP.name: nn.Linear(in_features, out_features = 1),
            Tasks.CHANGE_DETECTION_SIZE.name: nn.Linear(in_features, out_features = 1),
            Tasks.CHANGE_DETECTION_CONJ.name: nn.Linear(in_features, out_features = 1),
        })

        return Classifier_Head_Map

    def _flatten_mem_params(self):
        if isinstance(self.Memory, nn.RNNBase):
                    self.Memory.flatten_parameters()



            

        


        