# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn import ReLU


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
        attn_output, _ = self.cross_attn(
            query=query_data,
            key=context_data,
            value=context_data,
            need_weights=False
        )
        output = self.norm1(
            query_data + self.dropout1(attn_output)
        )
        ffn_output = self.linear2(
            self.ff_dropout(
                self.activation(
                    self.linear1(output)
                )
            )
        )
        return self.norm2(
            output + self.dropout2(ffn_output)
        )


class CoAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(CoAttentionLayer, self).__init__()
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
        self.fusion_projection = nn.Linear(d_model * 2, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)

    def forward(self, data_a, data_b):
        a_data = self.a_from_b(
            query_data=data_a,
            context_data=data_b
        )
        b_data = self.b_from_a(
            query_data=data_b,
            context_data=data_a
        )
        fusion_data = torch.cat(
            (a_data, b_data),
            dim=-1
        )
        fusion_data = self.fusion_projection(fusion_data)
        return self.fusion_norm(fusion_data)


class MTEGDRP(torch.nn.Module):
    """
    在原 MTEGDRP 多组学分支基础上改为癌症亚型分类。

    多组学分支使用四层渐进式双向 Co-Attention：
    1. 三种组学分别经过一层 Transformer Encoder
    2. 突变与甲基化经过两层双向 Co-Attention
    3. 融合表示与 mRNA 再经过两层双向 Co-Attention
    4. 最终融合表示平均池化并映射为 128 维
    5. 保留原来的癌症亚型分类全连接层与三返回值接口

    删除：
    1. 药物 MAT / GCN / GIN / EGNN 分支
    2. 原 Transformer Decoder

    output_dim 由具体癌种的类别数决定。
    """

    def __init__(
            self,
            output_dim=2,
            num_features_xd=78,
            ge_features_dim=128,
            num_features_xt=25,
            embed_dim=128,
            mut_feature_dim=128,
            meth_feature_dim=128,
            connect_dim=128,
            dropout=0.2
    ):
        super(MTEGDRP, self).__init__()

        # 保留原构造函数参数，避免影响原有调用方式。
        # 当前分类模型的三组学输入均为 KPCA 后的 128 维。
        self.output_dim = output_dim

        # 多组学 Transformer 参数：每个 KPCA 分量作为一个 token
        # 输入 [batch, 128] -> [batch, 128, 1] -> [batch, 128, 32]
        omics_model_dim = 32
        omics_nhead = 4
        omics_ff_dim = 128

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

        self.ca_layer_1 = CoAttentionLayer(
            d_model=omics_model_dim,
            nhead=omics_nhead,
            dim_feedforward=omics_ff_dim,
            dropout=dropout
        )
        self.ca_layer_2 = CoAttentionLayer(
            d_model=omics_model_dim,
            nhead=omics_nhead,
            dim_feedforward=omics_ff_dim,
            dropout=dropout
        )
        self.ca_layer_3 = CoAttentionLayer(
            d_model=omics_model_dim,
            nhead=omics_nhead,
            dim_feedforward=omics_ff_dim,
            dropout=dropout
        )
        self.ca_layer_4 = CoAttentionLayer(
            d_model=omics_model_dim,
            nhead=omics_nhead,
            dim_feedforward=omics_ff_dim,
            dropout=dropout
        )
        self.omics_output_projection = Linear(
            omics_model_dim,
            connect_dim
        )
        self.mut_output_projection = Linear(
            omics_model_dim,
            connect_dim
        )
        fusion_input_dim = connect_dim

        # 按原模型保留全连接层结构，仅修改第一层输入维度
        self.fc1_all = Linear(
            fusion_input_dim,
            1024
        )

        self.fc2_all = Linear(
            1024,
            512
        )

        self.fc3_all = Linear(
            512,
            256
        )

        self.fc4_all = Linear(
            256,
            128
        )

        # 回归输出改为癌症亚型分类 logits
        self.out = Linear(
            connect_dim,
            output_dim
        )

        # 保留原激活函数和 dropout 设置
        self.relu = ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        ge_data = data.target_ge
        meth_data = data.target_meth
        mut_data = data.target_mut

        # mRNA：
        # [batch, 128] -> [batch, 128, 1] -> [batch, 128, 32]
        ge_data = self.ge_value_projection(
            ge_data.unsqueeze(-1)
        )

        ge_data = self.conv_ge_1(
            ge_data
        )

        # 突变：
        # [batch, 128] -> [batch, 128, 1] -> [batch, 128, 32]
        mut_data = self.mut_value_projection(
            mut_data.unsqueeze(-1)
        )

        mut_data = self.conv_mut_1(
            mut_data
        )

        # 保留原来的 128 维突变输出
        mut_output = self.mut_output_projection(
            mut_data.mean(dim=1)
        )

        # 甲基化：
        # [batch, 128] -> [batch, 128, 1] -> [batch, 128, 32]
        meth_data = self.meth_value_projection(
            meth_data.unsqueeze(-1)
        )

        meth_data = self.conv_meth_1(
            meth_data
        )

        fusion_1 = self.ca_layer_1(
            mut_data,
            meth_data
        )
        fusion_2 = self.ca_layer_2(
            fusion_1,
            meth_data
        )
        fusion_3 = self.ca_layer_3(
            fusion_2,
            ge_data
        )
        fusion_4 = self.ca_layer_4(
            fusion_3,
            ge_data
        )
        omics_data = self.omics_output_projection(
            fusion_4.mean(dim=1)
        )
        concat_data = omics_data

        # 原 Decoder 已删除，直接进入原来的全连接层
        concat_data = self.fc1_all(
            concat_data
        )

        concat_data = self.relu(
            concat_data
        )

        concat_data = self.dropout(
            concat_data
        )

        concat_data = self.fc2_all(
            concat_data
        )

        concat_data = self.relu(
            concat_data
        )

        concat_data = self.dropout(
            concat_data
        )

        concat_data = self.fc3_all(
            concat_data
        )

        concat_data = self.relu(
            concat_data
        )

        concat_data = self.dropout(
            concat_data
        )

        concat_data = self.fc4_all(
            concat_data
        )

        concat_data = self.relu(
            concat_data
        )

        concat_data = self.dropout(
            concat_data
        )

        # CrossEntropyLoss 直接接收 logits，因此这里不做 sigmoid/softmax
        out = self.out(
            concat_data
        )

        # 为尽量保留原训练代码的三返回值形式：
        # 第二个返回值为融合后的多组学表示，第三个仍为突变 128 维表示。
        return out, omics_data, mut_output
