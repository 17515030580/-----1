# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn import ReLU


class MTEGDRP(torch.nn.Module):
    """
    在原 MTEGDRP 多组学分支基础上改为癌症亚型分类。

    保留：
    1. 三种组学分别三层 Transformer Encoder
    2. 甲基化、突变分别以 mRNA 为 Query 的 Cross-Attention
    3. 两个门控残差融合
    4. 融合后的 mRNA 再经过一层 Transformer
    5. fused mRNA 128维 + methylation 64维 + mutation 64维
    6. 原来的全连接层 256 -> 1024 -> 512 -> 256 -> 128

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

        # 单组组学数据特征 -- GE
        self.ge_value_projection = Linear(
            1,
            omics_model_dim
        )

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

        self.EncoderLayer_ge_2 = nn.TransformerEncoderLayer(
            d_model=omics_model_dim,
            nhead=omics_nhead,
            dim_feedforward=omics_ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.conv_ge_2 = nn.TransformerEncoder(
            self.EncoderLayer_ge_2,
            1
        )

        self.EncoderLayer_ge_3 = nn.TransformerEncoderLayer(
            d_model=omics_model_dim,
            nhead=omics_nhead,
            dim_feedforward=omics_ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.conv_ge_3 = nn.TransformerEncoder(
            self.EncoderLayer_ge_3,
            1
        )

        # 单组组学数据特征 -- MUT
        self.mut_value_projection = Linear(
            1,
            omics_model_dim
        )

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

        self.EncoderLayer_mut_2 = nn.TransformerEncoderLayer(
            d_model=omics_model_dim,
            nhead=omics_nhead,
            dim_feedforward=omics_ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.conv_mut_2 = nn.TransformerEncoder(
            self.EncoderLayer_mut_2,
            1
        )

        self.EncoderLayer_mut_3 = nn.TransformerEncoderLayer(
            d_model=omics_model_dim,
            nhead=omics_nhead,
            dim_feedforward=omics_ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.conv_mut_3 = nn.TransformerEncoder(
            self.EncoderLayer_mut_3,
            1
        )

        # 单组组学数据特征 -- METH
        self.meth_value_projection = Linear(
            1,
            omics_model_dim
        )

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

        self.EncoderLayer_meth_2 = nn.TransformerEncoderLayer(
            d_model=omics_model_dim,
            nhead=omics_nhead,
            dim_feedforward=omics_ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.conv_meth_2 = nn.TransformerEncoder(
            self.EncoderLayer_meth_2,
            1
        )

        self.EncoderLayer_meth_3 = nn.TransformerEncoderLayer(
            d_model=omics_model_dim,
            nhead=omics_nhead,
            dim_feedforward=omics_ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.conv_meth_3 = nn.TransformerEncoder(
            self.EncoderLayer_meth_3,
            1
        )

        # 甲基化和突变分别通过交叉注意力融入 mRNA
        # 三种组学的输入、输出形状均为 [batch, 128, 32]
        self.meth_to_ge_cross_attention = nn.MultiheadAttention(
            embed_dim=omics_model_dim,
            num_heads=omics_nhead,
            dropout=dropout,
            batch_first=True
        )

        self.mut_to_ge_cross_attention = nn.MultiheadAttention(
            embed_dim=omics_model_dim,
            num_heads=omics_nhead,
            dropout=dropout,
            batch_first=True
        )

        # 门控：分别控制甲基化和突变信息融入 mRNA 的比例
        self.meth_gate = nn.Sequential(
            Linear(
                omics_model_dim * 2,
                omics_model_dim
            ),
            nn.Sigmoid()
        )

        self.mut_gate = nn.Sequential(
            Linear(
                omics_model_dim * 2,
                omics_model_dim
            ),
            nn.Sigmoid()
        )

        # 融合后的 mRNA 再经过一层 Transformer
        self.EncoderLayer_ge_fusion = nn.TransformerEncoderLayer(
            d_model=omics_model_dim,
            nhead=omics_nhead,
            dim_feedforward=omics_ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.conv_ge_fusion = nn.TransformerEncoder(
            self.EncoderLayer_ge_fusion,
            1
        )

        # 融合后的 mRNA 平均池化后映射为 128 维
        self.ge_output_projection = Linear(
            omics_model_dim,
            connect_dim
        )

        # 三层 Transformer 后的甲基化表示映射为 64 维
        self.meth_concat_projection = Linear(
            omics_model_dim,
            64
        )

        # 三层 Transformer 后的突变表示映射为 64 维
        self.mut_concat_projection = Linear(
            omics_model_dim,
            64
        )

        # 保留原来的 128 维突变特征作为模型第三个返回值
        self.mut_output_projection = Linear(
            omics_model_dim,
            connect_dim
        )

        # 删除药物分支后：融合 mRNA 128 + 甲基化 64 + 突变 64 = 256
        fusion_input_dim = (
            connect_dim
            + 64
            + 64
        )

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

        ge_data = self.conv_ge_2(
            ge_data
        )

        ge_data = self.conv_ge_3(
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

        mut_data = self.conv_mut_2(
            mut_data
        )

        mut_data = self.conv_mut_3(
            mut_data
        )

        # 保留原来的 128 维突变输出
        mut_output = self.mut_output_projection(
            mut_data.mean(dim=1)
        )

        # 三层 Transformer 后的突变表示映射为 64 维，用于最终拼接
        mut_concat_output = self.mut_concat_projection(
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

        meth_data = self.conv_meth_2(
            meth_data
        )

        meth_data = self.conv_meth_3(
            meth_data
        )

        # 三层 Transformer 后的甲基化表示映射为 64 维，用于最终拼接
        meth_output = self.meth_concat_projection(
            meth_data.mean(dim=1)
        )

        # 甲基化和突变分别以 mRNA 为 Query 进行交叉注意力
        meth_context, _ = self.meth_to_ge_cross_attention(
            query=ge_data,
            key=meth_data,
            value=meth_data,
            need_weights=False
        )

        mut_context, _ = self.mut_to_ge_cross_attention(
            query=ge_data,
            key=mut_data,
            value=mut_data,
            need_weights=False
        )

        # 门控融合：保留 mRNA 主分支，并控制两种辅助组学的注入强度
        meth_gate = self.meth_gate(
            torch.cat(
                (
                    ge_data,
                    meth_context
                ),
                dim=-1
            )
        )

        mut_gate = self.mut_gate(
            torch.cat(
                (
                    ge_data,
                    mut_context
                ),
                dim=-1
            )
        )

        ge_data = (
            ge_data
            + meth_gate * meth_context
            + mut_gate * mut_context
        )

        # 融合后的 mRNA 再经过 Transformer
        ge_data = self.conv_ge_fusion(
            ge_data
        )

        # 对 128 个 token 做平均池化，再映射为 [batch, 128]
        ge_data = self.ge_output_projection(
            ge_data.mean(dim=1)
        )

        # 最终拼接：融合 mRNA 128 + 甲基化 64 + 突变 64 = 256
        concat_data = torch.cat(
            (
                ge_data,
                meth_output,
                mut_concat_output
            ),
            dim=1
        )

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
        # 第二个返回值改为融合后的 mRNA 表示，第三个仍为突变 128 维表示。
        return out, ge_data, mut_output
