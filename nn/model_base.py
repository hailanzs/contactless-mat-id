
import torch
import torch.nn as nn

class MyLinear(nn.Module):
    """
    Custom linear module with optional dropout and batch normalization.

    Parameters:
    - p: Dropout probability.
    - in_features: Number of input features.
    - out_features: Number of output features.
    - in_channels: Number of input channels.
    - b: Boolean flag indicating whether to apply batch normalization.

    Methods:
    - __init__(self, p, in_features, out_features, in_channels, b): Initializes the module.
    - forward(self, x): Performs the forward pass of the module.

    """

    def __init__(self, p=0.5, in_features=128, out_features=128, in_channels=1, b=1):
        super(MyLinear, self).__init__()
        self.p = p
        self.b = b
        self.dropout = nn.Dropout(p=self.p, inplace=False)  # Dropout layer with the specified dropout probability
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)  # Linear transformation
        self.bn = nn.BatchNorm1d(num_features=in_channels)  # Batch normalization

    def forward(self, x):
        """
        Performs the forward pass of the module.

        Parameters:
        - x: Input tensor.

        Returns:
        - x: Output tensor after linear transformation, batch normalization, and dropout (if enabled).
        """

        x = self.linear(x)  # Apply linear transformation
        if self.b:
            x = self.bn(x)  # Apply batch normalization if enabled
        if self.p:
            x = self.dropout(x)  # Apply dropout if enabled
        return x

        
class SimpleModel(nn.Module):
    """
    Simple model class for a neural network model.

    Parameters:
    - input_size_fft: Input size for FFT layers.
    - input_size_mrf: Input size for MRF layers.
    - input_size_phase: Input size for phase layers.
    - output_size: Output size of the model.
    - fft_channels: Number of channels for FFT layers.
    - first_channels: Number of channels for the first layer.
    - p: Dropout probability.
    - device: Device index.

    Attributes:
    - first_channels: Number of channels for the first layer.
    - input_size_fft: Input size for FFT layers.
    - input_size_mrf: Input size for MRF layers.
    - input_size_phase: Input size for phase layers.
    - output_size: Output size of the model.
    - fft_layers: List to store FFT layers.
    - mrf_layers: List to store MRF layers.
    - damp_layers: List to store damp layers.
    - final_layers: List to store final layers.
    - fft_final_layers: List to store final layers for FFT processing.
    - mrf_final_layers: List to store final layers for MRF processing.
    - phase_final_layers: List to store final layers for phase processing.
    - kernel_size: Kernel size for convolutional layers.
    - padding: Padding size for convolutional layers.
    - base_size: Base size for the model.
    - loss: Loss function.
    - stride: Stride value for unfolding.
    - unfold: Unfold operation.
    - folded_damp_size: Size of folded damp layers.
    - folded_damp_channels: Number of channels for folded damp layers.
    - fft_channels: Number of channels for FFT layers.
    - p: Dropout probability.
    - device: Cuda device index.
    - conv_k:Convolution kernel size
    - conv_p: Convolution padding size
    - stride: Convolution stride
    - final_fft_features_size: Size of final FFT features
    - final_damp_features_size: Size of final damp features
    """

    def __init__(self, input_size_fft, input_size_mrf, input_size_phase, output_size, fft_channels, first_channels, p=0, device=0):
        super(SimpleModel, self).__init__()
        self.first_channels = first_channels
        self.input_size_fft = input_size_fft
        self.input_size_mrf = input_size_mrf
        self.input_size_phase = input_size_phase
        self.output_size = output_size
        self.fft_layers = []  # 240 x 1
        self.mrf_layers = []  # 5 x 1
        self.damp_layers = []  # 3250 x 1
        self.final_layers = []
        self.fft_final_layers = []
        self.mrf_final_layers = []
        self.phase_final_layers = []
        self.kernel_size = 2
        self.padding = 0
        self.base_size = 128
        self.loss = nn.CrossEntropyLoss()
        self.stride = 250
        self.unfold = nn.Unfold(kernel_size=(1, 375), stride=(1, self.stride))
        self.folded_damp_size = 250
        self.folded_damp_channels = 8
        self.fft_channels = fft_channels
        self.p = p  # dropout probability
        self.device = device

        
        self.conv_k = 3  # Convolution kernel size
        self.conv_p = 1  # Convolution padding size
        self.stride = 1  # Convolution stride
        self.final_fft_features_size = 286  # Size of final FFT features
        self.final_damp_features_size = 125  # Size of final damp features

        self.fft_layers.extend([
            # X by 286
            nn.Conv1d(in_channels=self.fft_channels, out_channels=self.first_channels, kernel_size=self.conv_k, padding=self.conv_p, stride=self.stride),
            nn.BatchNorm1d(num_features=self.first_channels),
            nn.ReLU(),

            # 8 by 286
            nn.Conv1d(in_channels=self.first_channels, out_channels=self.first_channels, kernel_size=self.conv_k, padding=self.conv_p, stride=self.stride),
            nn.BatchNorm1d(num_features=self.first_channels),
            nn.ReLU(),

            # 8 by 286
            nn.Conv1d(in_channels=self.first_channels, out_channels=self.first_channels, kernel_size=self.conv_k, padding=self.conv_p, stride=self.stride),
            nn.BatchNorm1d(num_features=self.first_channels),
            nn.ReLU(),

            # 8 by 286
            nn.Conv1d(in_channels=self.first_channels, out_channels=self.first_channels, kernel_size=self.conv_k, padding=self.conv_p, stride=self.stride),
            nn.BatchNorm1d(num_features=self.first_channels),
            nn.ReLU(),

            # 8 by 286
            nn.Conv1d(in_channels=self.first_channels, out_channels=self.first_channels, kernel_size=self.conv_k, padding=self.conv_p, stride=self.stride),
            nn.BatchNorm1d(num_features=self.first_channels),
            nn.ReLU(),

            # 8 by 286
            nn.Conv1d(in_channels=self.first_channels, out_channels=self.first_channels, kernel_size=self.conv_k, padding=self.conv_p, stride=self.stride),
            nn.BatchNorm1d(num_features=self.first_channels),
            nn.ReLU(),

            # 8 by 286
        ])

        self.mrf_layers.extend([
            MyLinear(p=self.p, in_features=self.input_size_mrf, out_features=self.base_size),
            nn.MaxPool1d(kernel_size=self.kernel_size, stride=1, padding=self.padding),
            nn.ReLU(),
            MyLinear(p=self.p, in_features=self.base_size-1, out_features=self.base_size),
            nn.MaxPool1d(kernel_size=self.kernel_size, stride=1, padding=self.padding),
            nn.ReLU(),
            MyLinear(p=self.p, in_features=self.base_size-1, out_features=self.base_size),
            nn.ReLU(),
        ])
        
        self.damp_layers.extend([
            
            # 8 by 125
            
            nn.Conv1d(in_channels=self.folded_damp_channels, out_channels=self.first_channels, kernel_size=self.conv_k, padding=self.conv_p, stride=self.stride),
            nn.BatchNorm1d(num_features=self.first_channels),
            nn.ReLU(),
            
            # 8 by 125
            
            nn.Conv1d(in_channels=self.first_channels, out_channels=self.first_channels, kernel_size=self.conv_k, padding=self.conv_p, stride=self.stride),
            nn.BatchNorm1d(num_features=self.first_channels),
            nn.ReLU(),
            
            # 8 by 125
            
            nn.Conv1d(in_channels=self.first_channels, out_channels=self.first_channels, kernel_size=self.conv_k, padding=self.conv_p, stride=self.stride),
            nn.BatchNorm1d(num_features=self.first_channels),
            nn.ReLU(),
            
            # 8 by 125
            
            nn.Conv1d(in_channels=self.first_channels, out_channels=self.first_channels, kernel_size=self.conv_k, padding=self.conv_p, stride=self.stride),
            nn.BatchNorm1d(num_features=self.first_channels),
            nn.ReLU(),
            
            # 8 by 125
            
            nn.Conv1d(in_channels=self.first_channels, out_channels=self.first_channels, kernel_size=self.conv_k, padding=self.conv_p, stride=self.stride),
            nn.BatchNorm1d(num_features=self.first_channels),
            nn.ReLU(),
            
            # 8 by 125

            nn.Conv1d(in_channels=self.first_channels, out_channels=self.first_channels, kernel_size=self.conv_k, padding=self.conv_p, stride=self.stride),
            nn.BatchNorm1d(num_features=self.first_channels),
            nn.ReLU(),
            
            # 8 by 125
            
        ])
    
        
        self.final_layers.extend([
            MyLinear(p=self.p, in_features=self.base_size*3, out_features=self.output_size),
            nn.ReLU(),
            nn.Linear(in_features=self.output_size, out_features=self.output_size)
        ])
        self.fft_final_layers.extend([
            MyLinear(p=self.p, in_features=self.final_fft_features_size, out_features=self.output_size),
            nn.ReLU(),
            nn.Linear(in_features=self.output_size, out_features=self.output_size)
        ])
        self.mrf_final_layers.extend([
            MyLinear(p=self.p, in_features=self.base_size, out_features=self.output_size),
            nn.ReLU(),
            nn.Linear(in_features=self.output_size, out_features=self.output_size)
        ])
        self.phase_final_layers.extend([
            MyLinear(p=self.p, in_features=self.final_damp_features_size, out_features=self.output_size),
            nn.ReLU(),
            nn.Linear(in_features=self.output_size, out_features=self.output_size)
        ])
        
        # Convert the list of FFT layers into a sequential container
        self.fft_layers = nn.Sequential(*self.fft_layers)

        # Convert the list of MRF layers into a sequential container
        self.mrf_layers = nn.Sequential(*self.mrf_layers)

        # Convert the list of dampening layers into a sequential container
        self.damp_layers = nn.Sequential(*self.damp_layers)

        # Convert the list of final layers into a sequential container
        self.final_layers = nn.Sequential(*self.final_layers)

        # Convert the list of FFT final layers into a sequential container
        self.fft_final_layers = nn.Sequential(*self.fft_final_layers)

        # Convert the list of MRF final layers into a sequential container
        self.mrf_final_layers = nn.Sequential(*self.mrf_final_layers)

        # Convert the list of phase final layers into a sequential container
        self.phase_final_layers = nn.Sequential(*self.phase_final_layers)

        # Define a linear layer for combining FFT features
        self.single_linear_layer = nn.Linear(in_features=self.final_fft_features_size, out_features=self.base_size)

        # Define a linear layer for combining dampening features
        self.single_linear_layer_damp = nn.Linear(in_features=self.final_damp_features_size, out_features=self.base_size)

        # Move the dampening linear layer to the specified device
        self.single_linear_layer_damp.to(self.device)

        # Move the FFT linear layer to the specified device
        self.single_linear_layer.to(self.device)

        
        
    def forward(self, x1, x2, x3):
        """
        Forward pass of the model.

        Parameters:
        - x1: Input tensor 1.
        - x2: Input tensor 2.
        - x3: Input tensor 3.

        Returns:
        - x: Output tensor after processing and combining x1, x2, and x3.
        - x1: Output tensor after processing x1.
        - x2: Output tensor after processing x2.
        - x3: Output tensor after processing x3.
        """

        # Process x1
        x1 = self.fft_layers(x1)  # Apply FFT layers to x1
        x1 = torch.sum(x1, dim=1, keepdim=True).to(self.device)  # Sum along the second dimension
        x1_to_combine = self.single_linear_layer(x1)  # Apply linear layer to x1

        # Process x2
        x2 = self.mrf_layers(x2)  # Apply MRF layers to x2

        # Process x3
        x3 = self.damp_layers(x3)  # Apply damp layers to x3
        x3 = torch.sum(x3, dim=1, keepdim=True).to(self.device)  # Sum along the second dimension
        x3_to_combine = self.single_linear_layer_damp(x3)  # Apply linear layer to x3

        # Combine x1, x2, x3
        x = torch.cat((x1_to_combine, x2, x3_to_combine), dim=1)  # Concatenate along the second dimension
        x = torch.reshape(x, shape=(x.shape[0], 1, -1))  # Reshape x
        x = self.final_layers(x)  # Apply final layers to x

        x1 = self.fft_final_layers(x1)  # Apply final layers to x1
        x2 = self.mrf_final_layers(x2)  # Apply final layers to x2
        x3 = self.phase_final_layers(x3)  # Apply final layers to x3

        return torch.squeeze(x), torch.squeeze(x1), torch.squeeze(x2), torch.squeeze(x3)