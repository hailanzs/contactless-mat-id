
import torch
import torch.nn as nn

class MyLinear(nn.Module):
    def __init__(self, p=0.5, in_features=128, out_features=128, in_channels=1, b=1):
        super(MyLinear, self).__init__()
        self.p = p
        self.b = b
        self.dropout = nn.Dropout(p=self.p, inplace=False)
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.bn = nn.BatchNorm1d(num_features=in_channels)
        
    def forward(self, x):
        x = self.linear(x)
        if self.b:
            x = self.bn(x)
        if self.p:
            x = self.dropout(x)
        return x
        
class SimpleModel(nn.Module):
    def __init__(self, input_size_fft, input_size_mrf, input_size_phase, output_size, fft_channels, first_channels, p=0, device=0):
        
        super(SimpleModel, self).__init__()
        self.first_channels = first_channels
        self.input_size_fft = input_size_fft
        self.input_size_mrf = input_size_mrf
        self.input_size_phase = input_size_phase
        self.output_size = output_size
        self.fft_layers = [] # 240 x 1
        self.mrf_layers = [] # 5 x 1
        self.damp_layers = [] # 3250 x 1
        self.final_layers = []
        self.fft_final_layers = []
        self.mrf_final_layers = []
        self.phase_final_layers = []
        self.kernel_size = 2
        self.padding = 0
        self.base_size = 128
        self.loss = nn.CrossEntropyLoss()
        self.stride= 250
        self.unfold = nn.Unfold(kernel_size=(1, 375), stride=(1, self.stride))
        self.folded_damp_size = 250
        self.folded_damp_channels = 8
        self.fft_channels = fft_channels
        self.p = p # dropout probability
        self.device = device
        
        self.conv_k = 3
        self.conv_p = 1
        self.stride = 1
        self.final_fft_features_size = 286
        self.final_damp_features_size = 125
        
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
        
        self.fft_layers = nn.Sequential(*self.fft_layers)
        self.mrf_layers = nn.Sequential(*self.mrf_layers)
        self.damp_layers = nn.Sequential(*self.damp_layers)
        self.final_layers = nn.Sequential(*self.final_layers)
        self.fft_final_layers = nn.Sequential(*self.fft_final_layers)
        self.mrf_final_layers = nn.Sequential(*self.mrf_final_layers)
        self.phase_final_layers = nn.Sequential(*self.phase_final_layers)
        
        self.single_linear_layer = nn.Linear(in_features=self.final_fft_features_size, out_features=self.base_size)
        self.single_linear_layer_damp = nn.Linear(in_features=self.final_damp_features_size, out_features=self.base_size)
        self.single_linear_layer_damp.to(self.device)
        self.single_linear_layer.to(self.device)
        
        
    def forward(self, x1, x2, x3):
        
        # process x1
        x1 = self.fft_layers(x1) 
        x1 = torch.sum(x1, dim=1, keepdim=True).to(self.device)
        x1_to_combine = self.single_linear_layer(x1)
        
        # process x2
        x2 = self.mrf_layers(x2)
        
        # process x3
        x3 = self.damp_layers(x3)
        x3 = torch.sum(x3, dim=1, keepdim=True).to(self.device)
        x3_to_combine = self.single_linear_layer_damp(x3)

        # combine x1, x2, x3
        x = torch.cat((x1_to_combine, x2,x3_to_combine), dim=1)
        x = torch.reshape(x, shape=(x.shape[0], 1, -1))
        x = self.final_layers(x)
        
        x1 = self.fft_final_layers(x1)
        x2 = self.mrf_final_layers(x2)
        x3 = self.phase_final_layers(x3)
            
        return torch.squeeze(x), torch.squeeze(x1), torch.squeeze(x2), torch.squeeze(x3)
        
        
        