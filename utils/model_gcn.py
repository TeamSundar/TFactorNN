import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GENConv
from torch_geometric.nn import global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(5)
        self.conv1 = GCNConv(8, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        #self.conv1 = GENConv(8, hidden_channels)
        #self.conv2 = GENConv(hidden_channels, hidden_channels)
        #self.conv3 = GENConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)
        
    def forward(self, x, edge_index, batch):
        # Get node embeeding
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        
        # Readout alayer
        x = global_mean_pool(x, batch)
        
        # Out layer
        # x = F.dropout(x, p=DROPOUT, training=self.training)
        x = self.lin(x)
        return torch.sigmoid(x)