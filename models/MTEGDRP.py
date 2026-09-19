"""
四层渐进式双向Co-Attention融合
# 三组学token嵌入→三组学分别经过1层Transformer→
# 第1层CA：突变↔甲基化→Fusion_1→
# 第2层CA：Fusion_1↔甲基化→Fusion_2→
# 第3层CA：Fusion_2↔mRNA→Fusion_3→
# 第4层CA：Fusion_3↔mRNA→Fusion_4→平均池化→128维多组学融合特征
# 4层CA参数互不共享；每个方向均使用Residual + LayerNorm + FFN + Residual + LayerNorm，层输出再做LayerNorm

__coding__: utf-8
__Author__: Liu Zhihan
__Time__: 2024/8/21 14:30
__File__: MTEGDRP.py
__remark__:
__Software__: PyCharm
"""
import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn import Sequential
from torch.nn import ReLU
import torch.nn.functional as F
from torch_geometric.nn import GINConv as GIN_layer
from torch_geometric.nn import GCNConv as GCN_layer
from torch_geometric.nn import GATConv as GAT_layer, BatchNorm, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp
from torch import nn, einsum
from einops import rearrange
from models.egnn_pytorch import EGNN

DIST_KERNELS = {
    'exp': {
        'fn': lambda t: torch.exp(-t),
        'mask_value_fn': lambda t: torch.finfo(t.dtype).max
    },
    'softmax': {
        'fn': lambda t: torch.softmax(t, dim=-1),
        'mask_value_fn': lambda t: -torch.finfo(t.dtype).max
    }
}


def exists(val):
    return val is not None


def default(val, d):
    return d if not exists(val) else val


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim=78, heads=6, dim_head=13, Lg=0.5, Ld=0.5, La=1, dist_kernel_fn='exp'):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.La = La
        self.Ld = Ld
        self.Lg = Lg

        self.dist_kernel_fn = dist_kernel_fn

    def forward(self, x, mask=None, adjacency_mat=None, distance_mat=None):
        h, La, Ld, Lg, dist_kernel_fn = self.heads, self.La, self.Ld, self.Lg, self.dist_kernel_fn

        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (h qkv d) -> b h n qkv d', h=h, qkv=3).unbind(dim=-2)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(distance_mat):
            distance_mat = rearrange(distance_mat, 'b i j -> b () i j')

        if exists(adjacency_mat):
            adjacency_mat = rearrange(adjacency_mat, 'b i j -> b () i j')

        if exists(mask):
            mask_value = torch.finfo(dots.dtype).max
            mask = mask[:, None, :, None] * mask[:, None, None, :]

            # mask attention
            dots.masked_fill_(~mask, -mask_value)

            if exists(adjacency_mat):
                adjacency_mat.masked_fill_(~mask, 0.)

        attn = dots.softmax(dim=-1)

        # sum contributions from adjacency and distance tensors
        attn = attn * La

        if exists(adjacency_mat):
            attn = attn + Lg * adjacency_mat

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class MAT(nn.Module):
    def __init__(
            self,
            *,
            dim_in=78,
            model_dim=78,
            dim_out=78,
            depth=1,
            heads=6,
            Lg=0.5,
            Ld=0.5,
            La=1,
            dist_kernel_fn='exp'
    ):
        super().__init__()

        self.embed_to_model = nn.Linear(dim_in, model_dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            layer = nn.ModuleList([
                Residual(PreNorm(model_dim, Attention(model_dim, heads=heads, Lg=Lg, Ld=Ld, La=La,
                                                      dist_kernel_fn=dist_kernel_fn))),
                Residual(PreNorm(model_dim, FeedForward(model_dim)))
            ])
            self.layers.append(layer)

        self.norm_out = nn.LayerNorm(model_dim)
        self.ff_out = FeedForward(model_dim, dim_out)

    def forward(
            self,
            x,
            mask=None,
            adjacency_mat=None,
            distance_mat=None
    ):
        x = self.embed_to_model(x)

        for (attn, ff) in self.layers:
            x = attn(
                x,
                mask=mask,
                adjacency_mat=adjacency_mat,
                distance_mat=distance_mat
            )
            x = ff(x)

        x = self.norm_out(x)
        x = x.mean(dim=-2)
        x = self.ff_out(x)
        return x


# Define the Transformer Decoder Layer
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, 2048)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(2048, d_model)
        self.activation = nn.ReLU()

    def forward(self, tgt, memory):
        tgt2 = self.self_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt


# Define the Transformer Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory):
        for layer in self.layers:
            tgt = layer(tgt, memory)
        return self.norm(tgt)


# 单个方向的Co-Attention Block：
# Query来自一个模态，Key/Value来自另一个模态
# Cross-Attention -> Residual + LayerNorm -> FFN -> Residual + LayerNorm
class CoAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(CoAttentionBlock, self).__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ff_dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.GELU()

    def forward(self, query_data, context_data):
        # Query来自query_data，Key和Value来自context_data
        attn_output, _ = self.cross_attn(
            query=query_data,
            key=context_data,
            value=context_data,
            need_weights=False
        )

        # Cross-Attention后的残差连接和LayerNorm
        output = self.norm1(
            query_data + self.dropout1(attn_output)
        )

        # FFN
        ffn_output = self.linear2(
            self.ff_dropout(
                self.activation(
                    self.linear1(output)
                )
            )
        )

        # FFN后的残差连接和LayerNorm
        output = self.norm2(
            output + self.dropout2(ffn_output)
        )

        return output


# 一层双向Co-Attention Layer：
# A <- B 和 B <- A 两个方向分别计算，再拼接并映射回32维
class CoAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(CoAttentionLayer, self).__init__()

        # 两个方向使用独立参数
        self.a_from_b = CoAttentionBlock(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.b_from_a = CoAttentionBlock(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # 两个方向的输出：[batch, 128, 32] + [batch, 128, 32]
        # 拼接为[batch, 128, 64]，再映射回[batch, 128, 32]
        self.fusion_projection = nn.Linear(d_model * 2, d_model)

        # 每层双向融合完成后再做一次LayerNorm，稳定层间传递
        self.fusion_norm = nn.LayerNorm(d_model)

    def forward(self, data_a, data_b):
        # A <- B
        a_conditioned_on_b = self.a_from_b(
            query_data=data_a,
            context_data=data_b
        )

        # B <- A
        b_conditioned_on_a = self.b_from_a(
            query_data=data_b,
            context_data=data_a
        )

        # 双向结果拼接后形成当前CA层的融合表示
        fusion_data = torch.cat(
            (a_conditioned_on_b, b_conditioned_on_a),
            dim=-1
        )
        fusion_data = self.fusion_projection(fusion_data)

        # 当前CA层双向融合后的LayerNorm
        fusion_data = self.fusion_norm(fusion_data)

        return fusion_data


class MTEGDRP(torch.nn.Module):
    def __init__(self, output_dim=1, num_features_xd=78,
                 ge_features_dim=128, num_features_xt=25, embed_dim=128,
                 mut_feature_dim=128, meth_feature_dim=128, connect_dim=128, dropout=0.2):
        super(MTEGDRP, self).__init__()

        self.mat = MAT(
            dim_in=78,
            model_dim=78,
            dim_out=78 * 2,
            depth=4,
            heads=6,
            Lg=0.5,
            Ld=0.5,
            La=1,
            dist_kernel_fn='exp')
        self.conv_gcn = GCN_layer(num_features_xd * 2, num_features_xd * 2)

        net1 = Sequential(Linear(num_features_xd * 2, num_features_xd * 2), ReLU(),
                          Linear(num_features_xd * 2, num_features_xd * 2))
        self.conv_gin1 = GIN_layer(net1)
        self.bn1 = torch.nn.BatchNorm1d(num_features_xd * 2)

        self.fc_mat_egnn = Linear(num_features_xd * 2, num_features_xd)
        self.drug_layer1 = EGNN(dim=78)
        self.drug_layer2 = EGNN(dim=78)
        self.drug_layer3 = EGNN(dim=156)
        self.fc_drug_jihe1 = Linear(78, 390)
        self.fc_drug_jihe2 = Linear(390, 156)

        self.fc3_drug = Linear(312, 156)

        self.fc1_drug = Linear(num_features_xd * 6, num_features_xd * 12)
        self.fc2_drug = Linear(num_features_xd * 12, num_features_xd * 6)

        # 多组学Transformer参数：每个KPCA分量作为一个token
        # 输入[batch, 128]先变为[batch, 128, 1]，再投影为[batch, 128, 32]
        omics_model_dim = 32
        omics_nhead = 4
        omics_ff_dim = 128

        # mRNA：只使用1层Transformer提取单组学内部特征
        self.ge_value_projection = Linear(1, omics_model_dim)
        self.EncoderLayer_ge_1 = nn.TransformerEncoderLayer(
            d_model=omics_model_dim,
            nhead=omics_nhead,
            dim_feedforward=omics_ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.conv_ge_1 = nn.TransformerEncoder(
            self.EncoderLayer_ge_1,
            1
        )

        # 突变：只使用1层Transformer提取单组学内部特征
        self.mut_value_projection = Linear(1, omics_model_dim)
        self.EncoderLayer_mut_1 = nn.TransformerEncoderLayer(
            d_model=omics_model_dim,
            nhead=omics_nhead,
            dim_feedforward=omics_ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.conv_mut_1 = nn.TransformerEncoder(
            self.EncoderLayer_mut_1,
            1
        )

        # 甲基化：只使用1层Transformer提取单组学内部特征
        self.meth_value_projection = Linear(1, omics_model_dim)
        self.EncoderLayer_meth_1 = nn.TransformerEncoderLayer(
            d_model=omics_model_dim,
            nhead=omics_nhead,
            dim_feedforward=omics_ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.conv_meth_1 = nn.TransformerEncoder(
            self.EncoderLayer_meth_1,
            1
        )

        # 四层渐进式双向Co-Attention
        # 每一层均为独立实例，因此四层之间完全不共享参数

        # 第1层：Mutation <-> Methylation -> Fusion_1
        self.ca_layer_1 = CoAttentionLayer(
            d_model=omics_model_dim,
            nhead=omics_nhead,
            dim_feedforward=omics_ff_dim,
            dropout=dropout
        )

        # 第2层：Fusion_1 <-> Methylation -> Fusion_2
        self.ca_layer_2 = CoAttentionLayer(
            d_model=omics_model_dim,
            nhead=omics_nhead,
            dim_feedforward=omics_ff_dim,
            dropout=dropout
        )

        # 第3层：Fusion_2 <-> mRNA -> Fusion_3
        self.ca_layer_3 = CoAttentionLayer(
            d_model=omics_model_dim,
            nhead=omics_nhead,
            dim_feedforward=omics_ff_dim,
            dropout=dropout
        )

        # 第4层：Fusion_3 <-> mRNA -> Fusion_4
        self.ca_layer_4 = CoAttentionLayer(
            d_model=omics_model_dim,
            nhead=omics_nhead,
            dim_feedforward=omics_ff_dim,
            dropout=dropout
        )

        # Fusion_4平均池化后从32维映射为128维多组学融合特征
        self.omics_output_projection = Linear(
            omics_model_dim,
            connect_dim
        )

        # 保留128维突变特征作为模型第三个返回值
        # 使用“突变1层Transformer后的表示”平均池化再映射为128维
        self.mut_output_projection = Linear(
            omics_model_dim,
            connect_dim
        )

        # 药物特征468维 + 多组学融合特征128维 = 596维
        fusion_input_dim = num_features_xd * 6 + connect_dim

        # 保留原来的Transformer Decoder及其调用方式，不修改后续模块
        self.decoder = TransformerDecoder(
            num_layers=4,
            d_model=fusion_input_dim,
            nhead=4,
            dropout=0.1
        )

        self.fc1_all = Linear(fusion_input_dim, 1024)
        self.fc2_all = Linear(1024, 512)
        self.fc3_all = Linear(512, 256)
        self.fc4_all = Linear(256, 128)
        self.out = Linear(connect_dim, output_dim)

        # 激活函数和正则化
        self.relu = ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        drug_poi_data, drug_edg_index, batch = data.x, data.edge_index, data.batch
        ge_data, meth_data, mut_data = data.target_ge, data.target_meth, data.target_mut
        coors = data.coordinates

        drug_data = torch.unsqueeze(drug_poi_data, 1)
        mat_drug_data = self.mat(drug_data)
        drug_data = self.conv_gcn(mat_drug_data, drug_edg_index)
        drug_data = self.relu(drug_data)
        drug_data = F.relu(self.conv_gin1(drug_data, drug_edg_index))
        drug_data = self.bn1(drug_data)
        drug_data = self.conv_gcn(drug_data, drug_edg_index)
        drug_data = self.relu(drug_data)

        egnn_list = []
        index = 0
        for data_len in data.c_size:
            data_len = int(data_len.item())
            temp_data = mat_drug_data[index:index + data_len]
            temp_data = self.fc_mat_egnn(temp_data).unsqueeze(0)
            temp_coors = coors[index:index + data_len].unsqueeze(0)
            temp_drug_data_jihe, temp_ehnncoors = self.drug_layer1(temp_data, temp_coors)
            temp_drug_data_jihe, temp_ehnncoors = self.drug_layer2(temp_drug_data_jihe, temp_ehnncoors)
            temp_drug_data_jihe = temp_drug_data_jihe.squeeze(0)
            egnn_list.append(temp_drug_data_jihe)
            index += data_len

        egnn_features = torch.cat(egnn_list, dim=0)

        drug_data_final = torch.cat((drug_data, egnn_features), dim=1)
        drug_data_final = torch.cat([gmp(drug_data_final, batch), gap(drug_data_final, batch)], dim=1)

        drug_data = self.relu(self.fc1_drug(drug_data_final))
        drug_data = self.dropout(drug_data)
        drug_data = self.fc2_drug(drug_data)

        # ==============================================================
        # 多组学部分：1层Transformer + 4层渐进式双向Co-Attention
        # ==============================================================

        # mRNA：[batch, 128] -> [batch, 128, 1] -> [batch, 128, 32]
        # 再经过1层Transformer提取mRNA内部特征
        ge_data = self.ge_value_projection(
            ge_data.unsqueeze(-1)
        )
        ge_data = self.conv_ge_1(ge_data)

        # 突变：[batch, 128] -> [batch, 128, 1] -> [batch, 128, 32]
        # 再经过1层Transformer提取突变内部特征
        mut_data = self.mut_value_projection(
            mut_data.unsqueeze(-1)
        )
        mut_data = self.conv_mut_1(mut_data)

        # 甲基化：[batch, 128] -> [batch, 128, 1] -> [batch, 128, 32]
        # 再经过1层Transformer提取甲基化内部特征
        meth_data = self.meth_value_projection(
            meth_data.unsqueeze(-1)
        )
        meth_data = self.conv_meth_1(meth_data)

        # 保留突变1层Transformer后的128维输出，维持原来的第三返回值接口
        mut_output = self.mut_output_projection(
            mut_data.mean(dim=1)
        )

        # 第1层双向CA：Mutation <-> Methylation
        # mut_data和meth_data均为[batch, 128, 32]
        # 输出Fusion_1：[batch, 128, 32]
        fusion_1 = self.ca_layer_1(
            mut_data,
            meth_data
        )

        # 第2层双向CA：Fusion_1 <-> Methylation
        # 输出Fusion_2：[batch, 128, 32]
        fusion_2 = self.ca_layer_2(
            fusion_1,
            meth_data
        )

        # 第3层双向CA：Fusion_2 <-> mRNA
        # 输出Fusion_3：[batch, 128, 32]
        fusion_3 = self.ca_layer_3(
            fusion_2,
            ge_data
        )

        # 第4层双向CA：Fusion_3 <-> mRNA
        # 输出Fusion_4：[batch, 128, 32]
        fusion_4 = self.ca_layer_4(
            fusion_3,
            ge_data
        )

        # Fusion_4对128个token做平均池化：[batch, 128, 32] -> [batch, 32]
        # 再映射为最终多组学融合特征：[batch, 128]
        omics_data = self.omics_output_projection(
            fusion_4.mean(dim=1)
        )

        # 药物特征[batch, 468] + 多组学融合特征[batch, 128]
        # 最终拼接为[batch, 596]
        concat_data = torch.cat(
            (drug_data, omics_data),
            dim=1
        )

        # ==============================================================
        # 以下Decoder和后续FC保持原来的代码不变
        # ==============================================================

        # 保留原来的Transformer Decoder及其调用方式
        concat_data = concat_data.unsqueeze(0)
        concat_data = self.decoder(concat_data, concat_data)
        concat_data = concat_data.squeeze(0)

        # 隐藏层
        concat_data = self.fc1_all(concat_data)
        concat_data = self.relu(concat_data)
        concat_data = self.dropout(concat_data)
        concat_data = self.fc2_all(concat_data)
        concat_data = self.relu(concat_data)
        concat_data = self.dropout(concat_data)
        concat_data = self.fc3_all(concat_data)
        concat_data = self.relu(concat_data)
        concat_data = self.dropout(concat_data)
        concat_data = self.fc4_all(concat_data)
        concat_data = self.relu(concat_data)
        concat_data = self.dropout(concat_data)
        out = self.out(concat_data)
        # out = self.sigmoid(out)
        # out = nn.Sigmoid()(out)
        return out, drug_data, mut_output # Remove batch dimension after processing