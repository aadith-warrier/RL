import torch

class Linear_Layer(torch.nn.Module):
    def __init__(self, in_nodes, out_nodes):
        super(Linear_Layer, self).__init__()
        self.fc = torch.nn.Linear(in_nodes, out_nodes)
        torch.nn.init.xavier_normal_(self.fc.weight)
        self.af = torch.nn.Tanh()

    def forward(self,x):
        return self.af(self.fc(x))
    
class MLP(torch.nn.Module):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()
        self.model = torch.nn.Sequential()

        for layer in layer_sizes:
            self.model.append(Linear_Layer(layer[0], layer[1]))
        
    def forward(self, x):
        return self.model(x)
