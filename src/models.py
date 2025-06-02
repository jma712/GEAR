import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, DeepGraphInfomax, JumpingKnowledge
from torch.nn.utils import spectral_norm

EPS = 1e-15


class GCN(nn.Module):
    """
    The GCN class implements a simple Graph Convolutional Network layer.
    It wraps a single GCNConv layer (with spectral normalization applied)
    that transforms input node features from nfeat dimensions to nhid dimensions.
    The forward method applies this layer to the input features and edge indices,
    returning the transformed node features.
    """
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(GCN, self).__init__()
        self.gc1 = spectral_norm(GCNConv(nfeat, nhid))

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        return x


class GIN(nn.Module):
    """
    The GIN class implements a Graph Isomorphism Network layer.

    It uses a multi-layer perceptron (MLP) as the update function inside a GINConv layer. 
    The MLP consists of two spectrally normalized linear layers, a ReLU activation,
    and batch normalization. 
    
    The forward method applies the GIN convolution to the input features and edge indices.
    """
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(GIN, self).__init__()

        self.mlp1 = nn.Sequential(
            spectral_norm(nn.Linear(nfeat, nhid)),
            nn.ReLU(),
            nn.BatchNorm1d(nhid),
            spectral_norm(nn.Linear(nhid, nhid)),
        )
        self.conv1 = GINConv(self.mlp1)

    def forward(self, x, edge_iJKdex):
        x = self.conv1(x, edge_index)
        return x


class JK(nn.Module):
    """
    The JK class implements a GCN-based model with Jumping Knowledge (JK) connections, 
    which aggregate information from multiple layers to improve representation power. 

    It stacks two spectrally normalized GCN layers, applies a ReLU activation after each,
    and collects the outputs. 

    The JumpingKnowledge module then aggregates these outputs (using the 'max' mode).
    The class also includes a custom weight initialization method for linear layers.
    
    """
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(JK, self).__init__()
        self.conv1 = spectral_norm(GCNConv(nfeat, nhid))
        self.convx= spectral_norm(GCNConv(nhid, nhid))
        self.jk = JumpingKnowledge(mode='max')
        self.transition = nn.Sequential(
            nn.ReLU(),
        )

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        xs = []
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        xs.append(x)
        for _ in range(1):
            x = self.convx(x, edge_index)
            x = self.transition(x)
            xs.append(x)
        x = self.jk(xs)
        return x


class SAGE(nn.Module):
    """
    The SAGE class implements a two-layer GraphSAGE model. 
    Each layer is a SAGEConv with mean aggregation and normalization. 
    Between the two layers, it applies a transition block consisting of ReLU, 
    batch normalization, and dropout. 
    Like JK, it includes a custom weight initialization for linear layers. 
    The forward method applies the first SAGE layer, the transition, and then 
    the second SAGE layer.
    """
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(SAGE, self).__init__()

        # Implemented spectral_norm in the sage main file
        # ~/anaconda3/envs/PYTORCH/lib/python3.7/site-packages/torch_geometric/nn/conv/sage_conv.py
        self.conv1 = SAGEConv(nfeat, nhid, normalize=True)
        self.conv1.aggr = 'mean'
        self.transition = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(nhid),
            nn.Dropout(p=dropout)
        )
        self.conv2 = SAGEConv(nhid, nhid, normalize=True)
        self.conv2.aggr = 'mean'

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        x = self.conv2(x, edge_index)
        return x


class Encoder_DGI(nn.Module):
    def __init__(self, nfeat, nhid):
        super(Encoder_DGI, self).__init__()
        self.hidden_ch = nhid
        self.conv = spectral_norm(GCNConv(nfeat, self.hidden_ch))
        self.activation = nn.PReLU()

    def corruption(self, x, edge_index):
        # corrupted features are obtained by row-wise shuffling of the original features
        # corrupted graph consists of the same nodes but located in different places
        return x[torch.randperm(x.size(0))], edge_index

    def summary(self, z, *args, **kwargs):
        return torch.sigmoid(z.mean(dim=0))

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.activation(x)
        return x


class GraphInfoMax(nn.Module):
    def __init__(self, enc_dgi):
        super(GraphInfoMax, self).__init__()
        self.dgi_model = DeepGraphInfomax(enc_dgi.hidden_ch, enc_dgi, enc_dgi.summary, enc_dgi.corruption)

    def forward(self, x, edge_index):
        pos_z, neg_z, summary = self.dgi_model(x, edge_index)
        return pos_z


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                base_model='sage', k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model
        if self.base_model == 'gcn':
            self.conv = GCN(in_channels, out_channels)
        elif self.base_model == 'gin':
            self.conv = GIN(in_channels, out_channels)
        elif self.base_model == 'sage':
            self.conv = SAGE(in_channels, out_channels)
        elif self.base_model == 'infomax':
            enc_dgi = Encoder_DGI(nfeat=in_channels, nhid=out_channels)
            self.conv = GraphInfoMax(enc_dgi=enc_dgi)
        elif self.base_model == 'jk':
            self.conv = JK(in_channels, out_channels)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.conv(x, edge_index)
        return x


class Classifier(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(Classifier, self).__init__()

        # Classifier projector
        self.fc1 = spectral_norm(nn.Linear(ft_in, nb_classes))

    def forward(self, seq):
        ret = self.fc1(seq)
        return ret


# Importing necessary modules from PyTorch and other libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm

# Defining the GraphCF class, which inherits from PyTorch's nn.Module
class GraphCF(torch.nn.Module):
    # Constructor method to initialize the class
    def __init__(self, encoder, args, num_class):
        # Call the parent class constructor
        super(GraphCF, self).__init__()
        
        # Store the encoder model (used for graph representation learning)
        self.encoder = encoder
        
        # Hidden size of the encoder's output
        self.hidden_size = args.hidden_size
        
        # Size of the projection layer's hidden dimension
        self.num_proj_hidden = args.proj_hidden
        
        # Number of classes for classification
        self.num_class = num_class

        # Define the projection layers (fc1 and fc2) for transforming embeddings
        self.fc1 = nn.Sequential(
             # Linear layer with spectral normalization
            spectral_norm(nn.Linear(self.hidden_size, self.num_proj_hidden)), 
             # Batch normalization for stability
            nn.BatchNorm1d(self.num_proj_hidden), 
             # Activation function (ReLU)
            nn.ReLU(inplace=True) 
        )
        self.fc2 = nn.Sequential(
            # Linear layer with spectral normalization
            spectral_norm(nn.Linear(self.num_proj_hidden, self.hidden_size)),  
            # Batch normalization
            nn.BatchNorm1d(self.hidden_size)  
        )

        # Define the prediction layers (fc3 and fc4) for embedding refinement
        self.fc3 = nn.Sequential(
            spectral_norm(nn.Linear(self.hidden_size, self.hidden_size)),  
            nn.BatchNorm1d(self.hidden_size),  # Batch normalization
            nn.ReLU(inplace=True)  # Activation function (ReLU)
        )
        self.fc4 = spectral_norm(nn.Linear(self.hidden_size, self.hidden_size))  # Final prediction layer

        # Define the classifier layer for final classification
        self.c1 = Classifier(ft_in=self.hidden_size, nb_classes=num_class)

        # Initialize weights for all layers in the model
        for m in self.modules():
            self.weights_init(m)

        # Reset parameters of the encoder
        self.reset_parameters()

    # Method to reset parameters of the encoder
    def reset_parameters(self):
        reset(self.encoder)

    # Method to initialize weights for layers
    def weights_init(self, m):
        if isinstance(m, nn.Linear):  # Check if the layer is a Linear layer
            torch.nn.init.xavier_uniform_(m.weight.data)  # Initialize weights using Xavier uniform
            if m.bias is not None:  # If the layer has a bias term
                m.bias.data.fill_(0.0)  # Initialize bias to zero

    # Forward method to compute node and subgraph representations
    def forward(self, x, edge_index, batch=None, index=None):
        r""" Return node and subgraph representations of each node before and after 
        being shuffled """
        # Compute node representations using the encoder
        hidden = self.encoder(x, edge_index)  # Output shape: (batch size x subgraph size) x hidden_size
        if index is None:  # If no specific index is provided
            return hidden  # Return the full hidden representation

        # Extract representations for specific nodes (center nodes)
        z = hidden[index]  # Shape: batch_size x hidden_size
        return z

    # Method for projection transformation
    def projection(self, z):
        z = self.fc1(z)  # Apply the first projection layer
        z = self.fc2(z)  # Apply the second projection layer
        return z

    # Method for prediction transformation
    def prediction(self, z):
        z = self.fc3(z)  # Apply the first prediction layer
        z = self.fc4(z)  # Apply the second prediction layer
        return z

    # Method for classification
    def classifier(self, z):
        return self.c1(z)  # Pass the embeddings through the classifier

    # Method to normalize embeddings
    def normalize(self, x):
        val = torch.norm(x, p=2, dim=1).detach()  # Compute L2 norm for each embedding
        x = x.div(val.unsqueeze(dim=1).expand_as(x))  # Normalize embeddings by their norms
        return x

    # Method to compute entropy-based divergence
    def D_entropy(self, x1, x2):
        x2 = x2.detach()  # Detach x2 from the computation graph
        return (-torch.max(F.softmax(x2), dim=1)[0]*torch.log(torch.max(F.softmax(x1), dim=1)[0])).mean()

    # Method to compute negative cosine similarity
    def D(self, x1, x2):
        return -F.cosine_similarity(x1, x2.detach(), dim=-1).mean()

    # Method to compute fairness metrics
    def fair_metric(self, pred, labels, sens):
        idx_s0 = sens==0  # Identify samples with sensitive attribute 0
        idx_s1 = sens==1  # Identify samples with sensitive attribute 1

        idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)  # Samples with sensitive attribute 0 and label 1
        idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)  # Samples with sensitive attribute 1 and label 1

        # Compute parity and equality metrics
        parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
        equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))

        return parity.item(), equality.item()

    # Method to predict class labels from embeddings
    def predict(self, emb):
        p1 = self.projection(emb)  # Apply projection transformation
        h1 = self.prediction(p1)  # Apply prediction transformation
        c1 = self.classifier(emb)  # Apply classification
        return c1