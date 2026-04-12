import torch.nn as nn
from .blocks import EdgeBlock, NodeBlock
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, SAGEConv

def build_mlp(in_size, hidden_size, out_size, lay_norm=True):
    # 2 hidden layers — matches DeepMind MeshGraphNets (num_layers=2, latent_size=128)
    module = nn.Sequential(
        nn.Linear(in_size,     hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, out_size)
    )
    if lay_norm: return nn.Sequential(module, nn.LayerNorm(normalized_shape=out_size))
    return module


class Encoder(nn.Module):

    def __init__(self,
                edge_input_size=128,
                node_input_size=128,
                hidden_size=128):
        super(Encoder, self).__init__()

        self.eb_encoder = build_mlp(edge_input_size, hidden_size, hidden_size)
        self.nb_encoder = build_mlp(node_input_size, hidden_size, hidden_size)

    def forward(self, graph):

        node_attr, edge_attr = graph.x, graph.edge_attr
        node_ = self.nb_encoder(node_attr)
        edge_ = self.eb_encoder(edge_attr)

        return Data(x=node_, edge_attr=edge_, edge_index=graph.edge_index)


class GnBlock(nn.Module):

    def __init__(self, hidden_size=128):

        super(GnBlock, self).__init__()

        eb_input_dim = 3 * hidden_size
        nb_input_dim = 2 * hidden_size
        nb_custom_func = build_mlp(nb_input_dim, hidden_size, hidden_size)
        eb_custom_func = build_mlp(eb_input_dim, hidden_size, hidden_size)

        self.eb_module = EdgeBlock(custom_func=eb_custom_func)
        self.nb_module = NodeBlock(custom_func=nb_custom_func)

    def forward(self, graph):

        x = graph.x.clone()
        edge_attr = graph.edge_attr.clone()

        graph = self.eb_module(graph)
        graph = self.nb_module(graph)

        x = x + graph.x
        edge_attr = edge_attr + graph.edge_attr

        return Data(x=x, edge_attr=edge_attr, edge_index=graph.edge_index)


class TNSBlock(nn.Module):
    """
    Transformer-based processor block.

    Replaces GnBlock's scatter-sum + MLP with multi-head dot-product attention
    (TransformerConv). Edge features from the Encoder go into the attention key,
    giving each head access to geometric/relational info. Edge features are NOT
    updated per-block — they remain fixed from the Encoder output.

    Aggregation: attention-weighted sum  Σ_j α_ij · W·h_j
    α_ij = softmax( (W_Q·h_i)ᵀ (W_K·h_j + W_E·e_ij) / √d_head )
    Output kept at hidden_size via concat=True + out_channels=hidden_size//heads.
    """
    def __init__(self, hidden_size: int = 128, heads: int = 4, dropout: float = 0.0):
        super(TNSBlock, self).__init__()
        assert hidden_size % heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by heads ({heads})"
        )
        self.conv = TransformerConv(
            in_channels=hidden_size,
            out_channels=hidden_size // heads,  # concat=True → output = heads*(h//heads) = h
            heads=heads,
            concat=True,
            beta=True,            # learned gate blending self-transform and attention
            edge_dim=hidden_size, # edge features enter attention key computation
            dropout=dropout,
            root_weight=True,
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, graph: Data) -> Data:
        x_in = graph.x
        x_new = self.conv(graph.x, graph.edge_index, graph.edge_attr)
        graph.x = x_in + self.norm(x_new)
        # edge_attr unchanged — TNS does not update edge features per block
        return graph


class SAGEBlock(nn.Module):
    """
    GraphSAGE-based processor block.

    Aggregation: MEAN of neighbor node features, concatenated with self-features.
    new_h = Linear( cat(h_v, MEAN(h_neighbors)) )  [optionally L2-normalized]

    vs GnBlock (sum aggregation):
      - MEAN normalizes by degree → robust to high-valence mesh nodes
      - Explicit self/neighbor separation → reduces over-smoothing across 15 layers
      - No edge feature updates → faster, scales to larger meshes
      - No edge features in message → edge info not used by processor (only encoder)
    """
    def __init__(self, hidden_size: int = 128, aggr: str = "mean", normalize: bool = True):
        super(SAGEBlock, self).__init__()
        self.conv = SAGEConv(
            in_channels=hidden_size,
            out_channels=hidden_size,
            aggr=aggr,
            normalize=normalize,  # L2-normalize output (SAGE paper default)
            root_weight=True,
            bias=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.act  = nn.ReLU()

    def forward(self, graph: Data) -> Data:
        x_in = graph.x
        x_new = self.act(self.conv(graph.x, graph.edge_index))
        graph.x = x_in + self.norm(x_new)
        # edge_attr unchanged
        return graph


class Decoder(nn.Module):

    def __init__(self, hidden_size=128, output_size=2):
        super(Decoder, self).__init__()
        self.decode_module = build_mlp(hidden_size, hidden_size, output_size, lay_norm=False)

    def forward(self, graph):
        return self.decode_module(graph.x)


class EncoderProcesserDecoder(nn.Module):

    def __init__(self, message_passing_num, node_input_size, edge_input_size,
                 hidden_size=128, output_size=2,
                 architecture: str = "gn",
                 tns_heads: int = 4,
                 tns_dropout: float = 0.0,
                 sage_aggr: str = "mean",
                 sage_normalize: bool = True):

        super(EncoderProcesserDecoder, self).__init__()

        self.architecture = architecture

        self.encoder = Encoder(edge_input_size=edge_input_size,
                               node_input_size=node_input_size,
                               hidden_size=hidden_size)

        if architecture == "gn":
            blocks = [GnBlock(hidden_size=hidden_size)
                      for _ in range(message_passing_num)]
        elif architecture == "tns":
            assert hidden_size % tns_heads == 0, (
                f"hidden_size ({hidden_size}) must be divisible by tns_heads ({tns_heads})"
            )
            blocks = [TNSBlock(hidden_size=hidden_size, heads=tns_heads, dropout=tns_dropout)
                      for _ in range(message_passing_num)]
        elif architecture == "sage":
            blocks = [SAGEBlock(hidden_size=hidden_size, aggr=sage_aggr, normalize=sage_normalize)
                      for _ in range(message_passing_num)]
        else:
            raise ValueError(
                f"Unknown architecture: '{architecture}'. Valid options: 'gn', 'tns', 'sage'."
            )

        self.processer_list = nn.ModuleList(blocks)

        self.decoder = Decoder(hidden_size=hidden_size, output_size=output_size)

    def forward(self, graph):

        graph = self.encoder(graph)
        for model in self.processer_list:
            graph = model(graph)
        decoded = self.decoder(graph)

        return decoded
