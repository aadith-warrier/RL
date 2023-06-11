import torch

class Conv_Layer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_Layer, self).__init__()
        self.layer = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        torch.nn.BatchNorm2d(out_channels),
                        torch.nn.ReLU())   
    def forward(self, x):
        return self.layer(x)     
    
class CNN(torch.nn.Module):
    def __init__(self, layer_sizes):
        super(CNN, self).__init__()
        self.model = torch.nn.Sequential()

        for layer in layer_sizes:
            self.model.append(Conv_Layer(layer[0], layer[1]))
        
    def forward(self, x):
        return self.model(x)