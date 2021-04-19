import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        assert not torch.any(torch.isnan(x)), "FFN input"
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        assert not torch.any(torch.isnan(output)), "FFN output"
        return output

class SGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

        # self.fc = nn.Linear(in_feats, out_feats, bias=False)
        # self.attn_fc = nn.Linear(2 * out_feats, 1, bias=False)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [edge_num, 2 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(edges):
         return {'m': edges.data['h']}

    def reduce_func(nodes):
        
        return {{'h': torch.sum(nodes.mailbox['m'], dim=1)}}

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)

        g.nodes[wnode_id].data['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        h = g.ndata.pop('h')
        return self.linear(h)[snode_id]

class WSGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(WSGCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            wnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
            snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
            wsedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 0) & (edges.dst["unit"] == 1))

            g.nodes[wnode_id].data['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata.pop('h')
            return self.linear(h)[snode_id]


class SWGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(SWGCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            wnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
            snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
            swedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 0) & (edges.dst["unit"] == 1))

            g.nodes[snode_id].data['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata.pop('h')
            return self.linear(h)[wnode_id]