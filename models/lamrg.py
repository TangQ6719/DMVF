import os

import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.Transformer import TransformerModel
from modules.text_encoder import TextEncoder, MHA_FF
from modules.base_cmn import BaseCMN

class _LAMRG(nn.Module):
    def __init__(self, args, tokenizer):
        super(_LAMRG, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = TransformerModel(args, tokenizer)
        self.proj = nn.Linear(args.num_labels, args.d_vf)

        self._init_weight(self.proj)

    @staticmethod
    def _init_weight(f):
        nn.init.kaiming_normal_(f.weight)
        f.bias.data.fill_(0)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images):
        att_feats_0, fc_feats_0, labels_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1, labels_1 = self.visual_extractor(images[:, 1])

        fc_feats = torch.mean(torch.stack([fc_feats_0, fc_feats_1]), dim=0)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        out_labels = torch.mean(torch.stack([labels_0, labels_1]), dim=0)
        return att_feats, fc_feats, out_labels

    def forward_mimic_cxr(self, images):
        att_feats, fc_feats, out_labels = self.visual_extractor(images)
        return att_feats, fc_feats, out_labels

    def forward(self, images, targets=None, labels=None, mode='train'):
        if self.args.dataset_name == 'iu_xray':
            att_feats, fc_feats, out_labels = self.forward_iu_xray(images)
        else:
            att_feats, fc_feats, out_labels = self.forward_mimic_cxr(images)

        label_feats = self.proj(out_labels).unsqueeze(1)
        att_feats = torch.cat((att_feats, label_feats), dim=1)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, opt=self.args, mode='sample')
        else:
            raise ValueError
        return output, out_labels


class LAMRGModel(_LAMRG):
    def __init__(self, args, tokenizer):
        super(LAMRGModel, self).__init__(args, tokenizer)

        self.m = nn.Parameter(torch.FloatTensor(1, args.num_labels, 40, args.d_vf))
        self.init_m()

        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def init_m(self):
        nn.init.normal_(self.m, 0, 1 / self.args.num_labels)

    def forward_iu_xray(self, images, targets=None, labels=None, mode='train'):
        att_feats_0, fc_feats_0, labels_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1, labels_1 = self.visual_extractor(images[:, 1])

        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        out_labels = labels_0

        bs, nf, d_f = att_feats.shape
        _, n_l, n_m, d_f = self.m.shape
        m = labels[:, :, None, None] * self.m.expand(bs, n_l, n_m, d_f)
        m = m.reshape(bs, -1, d_f)
        att_feats = torch.cat((att_feats, m), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output, out_labels

    def forward_mimic_cxr(self, images, targets=None, labels=None, mode='train'):

        att_feats, fc_feats, out_labels = self.visual_extractor(images)

        bs, nf, d_f = att_feats.shape
        _, n_l, n_m, d_f = self.m.shape
        m = labels[:, :, None, None] * self.m.expand(bs, n_l, n_m, d_f)
        m = m.reshape(bs, -1, d_f)
        att_feats = torch.cat((att_feats, m), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output, out_labels


class BasicModel(LAMRGModel):
    def forward_iu_xray(self, images, targets=None, labels=None, mode='train'):
        att_feats_0, fc_feats_0, labels_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1, labels_1 = self.visual_extractor(images[:, 1])

        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        out_labels = labels_0

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output, out_labels

    def forward_mimic_cxr(self, images, targets=None, labels=None, mode='train'):

        att_feats, fc_feats, out_labels = self.visual_extractor(images)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output, out_labels


class LAMRGModel_v7(_LAMRG):
    """直接将visual_extractor输出的label concat到visual feature后"""
    def __init__(self, args, tokenizer):
        super(LAMRGModel_v7, self).__init__(args, tokenizer)

    def forward(self, images, targets=None, labels=None, mode='train'):
        if self.args.dataset_name == 'iu_xray':
            att_feats, fc_feats, out_labels = self.forward_iu_xray(images)
        else:
            att_feats, fc_feats, out_labels = self.forward_mimic_cxr(images)

        label_feats = self.proj(out_labels).unsqueeze(1)
        att_feats = torch.cat((att_feats, label_feats), dim=1)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, opt=self.args, mode='sample')
        else:
            raise ValueError
        return output, out_labels


class LAMRGModel_v8(_LAMRG):
    def __init__(self, args, tokenizer):
        super(LAMRGModel_v8, self).__init__(args, tokenizer)
        self.args = args
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_layers = args.num_layers
        self.num_labels = args.num_labels
        self.tgt_vocab = len(tokenizer.idx2token) + 1
        self.h = args.num_heads
        self.num_slots = args.num_slots
        self.d_vf = args.d_vf
        self.dropout = args.dropout

        self.txt_encoder = TextEncoder(self.d_model, self.d_ff, self.num_layers, self.tgt_vocab, self.num_labels,
                                       self.h, self.dropout)
        self.prior_memory = MHA_FF(self.d_model, self.d_ff, self.h, self.dropout)
        self.select_prior = MHA_FF(self.d_model, self.d_ff, self.h, self.dropout)
        self.memory = self.init_memory()

        self.proj_label = nn.Linear(args.num_labels, args.d_model)
        self.proj_att = nn.Linear(args.d_vf, args.d_model)
        self.proj_feat = nn.Linear(args.d_model, args.d_vf)
        self.init_weight_()

    def init_weight_(self):
        nn.init.kaiming_normal_(self.proj_label.weight)
        self.proj_label.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.proj_att.weight)
        self.proj_att.bias.data.fill_(0)

    def init_memory(self):
        memory = nn.Parameter(torch.eye(self.num_slots, device='cuda').unsqueeze(0))
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((1, self.num_slots, diff), device='cuda')
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]
        return memory

    def forward(self, images, targets=None, labels=None, mode='train'):
        if self.args.dataset_name == 'iu_xray':
            att_feats, fc_feats, out_labels = self.forward_iu_xray(images)
        else:
            att_feats, fc_feats, out_labels = self.forward_mimic_cxr(images)

        bsz = att_feats.shape[0]
        memory = self.memory.expand(bsz, self.num_slots, self.d_model)
        if mode == 'train':
            txt_feats, z_txt, txt_labels = self.txt_encoder(targets)
            memory = self.prior_memory(memory, txt_feats)

        label_feats = self.proj_label(out_labels).unsqueeze(1)
        prior = self.select_prior(label_feats, memory)
        att_feats = torch.cat((att_feats, self.proj_feat(prior)), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, opt=self.args, mode='sample')
        else:
            raise ValueError
        return output, out_labels


class LAMRGModel_v9(_LAMRG):
    def __init__(self, args, tokenizer):
        super(LAMRGModel_v9, self).__init__(args, tokenizer)
        self.args = args
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_layers = args.num_layers
        self.num_labels = args.num_labels
        self.tgt_vocab = len(tokenizer.idx2token) + 1
        self.h = args.num_heads
        self.num_slots = args.num_slots
        self.d_vf = args.d_vf
        self.dropout = args.dropout

        self.txt_encoder = TextEncoder(self.d_model, self.d_ff, self.num_layers, self.tgt_vocab, self.num_labels,
                                       self.h, self.dropout)
        self.prior_memory = MHA_FF(self.d_model, self.d_ff, self.h, self.dropout)
        self.select_prior = MHA_FF(self.d_model, self.d_ff, self.h, self.dropout)
        self.memory = self.init_memory()

        self.linear_mem = nn.Linear(self.d_model, self.d_model)
        self.linear_label = nn.Linear(args.num_labels, args.d_model)
        self.linear_feat = nn.Linear(args.d_model, args.d_vf)
        self.linear_fcfeat = nn.Linear(args.d_vf, args.d_model)
        self.init_weight_()

    def init_weight_(self):
        nn.init.kaiming_normal_(self.linear_mem.weight)
        self.linear_mem.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.linear_label.weight)
        self.linear_label.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.linear_feat.weight)
        self.linear_feat.bias.data.fill_(0)

    def init_memory(self):
        memory = nn.Parameter(torch.eye(self.num_slots, device='cuda').unsqueeze(0))
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((1, self.num_slots, diff), device='cuda')
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]
        return memory

    def forward(self, images, targets=None, labels=None, mode='train'):
        if self.args.dataset_name == 'iu_xray':
            att_feats, fc_feats, vis_labels = self.forward_iu_xray(images)
        else:
            att_feats, fc_feats, vis_labels = self.forward_mimic_cxr(images)

        z_img = self.linear_fcfeat(fc_feats)

        bsz = att_feats.shape[0]
        # memory = self.linear_mem(self.memory).expand(bsz, -1, -1)
        memory = self.memory.expand(bsz, -1, -1)
        if mode == 'train':
            txt_feats, z_txt, txt_labels = self.txt_encoder(targets)
            memory = self.prior_memory(memory, txt_feats)

        label_feats = self.linear_label(vis_labels).unsqueeze(1)
        prior = self.select_prior(label_feats, memory)
        att_feats = torch.cat((att_feats, self.linear_feat(prior)), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return (output, vis_labels, txt_labels, z_img, z_txt)
        elif mode == 'sample':
            # ipdb.set_trace()
            output, _ = self.encoder_decoder(fc_feats, att_feats, opt=self.args, mode='sample')
            return output, vis_labels
        else:
            raise ValueError


class LAMRGModel_v10(LAMRGModel_v9):
    def forward(self, images, targets=None, labels=None, mode='train'):
        if self.args.dataset_name == 'iu_xray':
            att_feats, fc_feats, vis_labels = self.forward_iu_xray(images)
        else:
            att_feats, fc_feats, vis_labels = self.forward_mimic_cxr(images)

        z_img = self.linear_fcfeat(fc_feats)

        bsz = att_feats.shape[0]
        memory = self.linear_mem(self.memory).expand(bsz, -1, -1)
        # memory = self.memory.expand(bsz, -1, -1)
        if mode == 'train':
            txt_feats, z_txt, txt_labels = self.txt_encoder(targets)
            memory = self.prior_memory(memory, txt_feats)

        label_feats = self.linear_label(vis_labels).unsqueeze(1)
        prior = self.select_prior(label_feats, memory)
        att_feats = torch.cat((att_feats, self.linear_feat(prior)), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return (output, vis_labels, txt_labels, z_img, z_txt)
        elif mode == 'sample':
            # ipdb.set_trace()
            output, _ = self.encoder_decoder(fc_feats, att_feats, opt=self.args, mode='sample')
            return output, vis_labels
        else:
            raise ValueError


class LAMRGModel_v11(LAMRGModel_v9):
    def __init__(self, args, tokenizer):
        super(LAMRGModel_v11, self).__init__(args, tokenizer)
        self.args = args
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_layers = args.num_layers
        self.num_labels = args.num_labels
        self.tgt_vocab = len(tokenizer.idx2token) + 1
        self.h = args.num_heads
        self.num_slots = args.num_slots
        self.d_vf = args.d_vf
        self.dropout = args.dropout

        self.txt_encoder = TextEncoder(self.d_model, self.d_ff, self.num_layers, self.tgt_vocab, self.num_labels,
                                       self.h, self.dropout)
        self.update_memory = MHA_FF(self.d_model, self.d_ff, self.h, self.dropout)
        self.select_prior = MHA_FF(self.d_model, self.d_ff, self.h, self.dropout)
        self.memory, self.mask = self.init_memory()

        self.linear_z = nn.Linear(args.d_vf, args.d_model)
        self.linear_label = nn.Linear(args.d_model, args.num_labels)
        self.query_mem = nn.Linear(self.d_model, self.d_model)
        self.linear_feat = nn.Linear(args.d_model, args.d_vf)
        self.init_weight()

    def init_weight(self):
        self._init_weight(self.linear_z)
        self._init_weight(self.linear_label)
        self._init_weight(self.query_mem)
        self._init_weight(self.linear_feat)

    def init_memory(self):
        memory = nn.Parameter(torch.eye(self.num_slots).unsqueeze(0))
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((1, self.num_slots, diff))
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]
        mask = torch.ones((self.num_slots, self.d_model))
        mask[:, self.num_slots:] = 0
        return memory, mask

    def forward(self, images, targets=None, labels=None, mode='train'):
        bsz = images.shape[0]

        ve = self.forward_iu_xray if self.args.dataset_name == 'iu_xray' else self.forward_mimic_cxr
        att_feats, avg_feats, _ = ve(images)
        z_img = self.linear_z(avg_feats)
        vis_labels = self.linear_label(z_img)

        memory = self.query_mem(self.memory.to(images)).expand(bsz, -1, -1)
        mask = self.mask.to(images).expand(bsz, -1, -1)
        if mode == 'train':
            txt_feats, z_txt, txt_labels = self.txt_encoder(targets)
            memory = self.update_memory(memory, txt_feats, mask)

        prior = self.select_prior(z_img, memory)
        att_feats = torch.cat((att_feats, self.linear_feat(prior)), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(avg_feats, att_feats, targets, mode='forward')
            return (output, vis_labels, txt_labels, z_img, z_txt)
        elif mode == 'sample':
            # ipdb.set_trace()
            output, _ = self.encoder_decoder(avg_feats, att_feats, opt=self.args, mode='sample')
            return output, vis_labels
        else:
            raise ValueError


# 项目模型
class LAMRGModel_v12(LAMRGModel_v9):
    def __init__(self, args, tokenizer):
        super(LAMRGModel_v12, self).__init__(args, tokenizer)
        self.args = args
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_layers = args.num_layers
        self.num_labels = args.num_labels
        self.tgt_vocab = len(tokenizer.idx2token) + 1
        self.h = args.num_heads
        self.num_slots = args.num_slots
        self.d_vf = args.d_vf
        self.num_box = args.num_box
        self.dropout = args.dropout

        # 初始化各个模块和网络层
        self.txt_encoder = TextEncoder(self.d_model, self.d_ff, self.num_layers, self.tgt_vocab, self.num_labels,
                                       self.h, self.dropout)
        self.update_memory = MHA_FF(self.d_model, self.d_ff, args.num_memory_heads, self.dropout)
        self.select_prior = MHA_FF(self.d_model, self.d_ff, args.num_memory_heads, self.dropout)
        self.memory, self.mask = self.init_memory()

        self.get_mem = nn.Linear(self.d_model, self.d_model)
        self.linear_z = nn.Linear(self.d_vf, self.d_model)
        self.linear_feat = nn.Linear(self.d_model, self.d_vf)

        self.classifier = nn.Linear(self.d_model, self.num_labels)
        self.embed_labels = nn.Linear(1, self.d_model)

        self.init_weight()

    def init_weight(self):
        # 初始化网络层的权重
        self._init_weight(self.linear_z)
        self._init_weight(self.get_mem)
        self._init_weight(self.linear_feat)
        self._init_weight(self.classifier)
        self._init_weight(self.embed_labels)

    def init_memory(self):
        # 初始化记忆矩阵和掩码
        memory = nn.Parameter(torch.eye(self.num_slots).unsqueeze(0))
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((1, self.num_slots, diff))
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]
        mask = torch.ones((self.num_slots, self.d_model))
        mask[:, self.num_slots:] = 0
        return memory, mask

    def forward(self, image_id, images, targets=None, labels=None, mode='train'):
        bsz = images.shape[0]
        # 根据数据集类型选择不同的前向传播函数
        ve = self.forward_iu_xray if self.args.dataset_name == 'iu_xray' else self.forward_mimic_cxr
        # 对数据集进行特征提取，得到注意力特征和池化特征
        att_feats, avg_feats, _ = ve(images)
        # 对特征进行处理，通过压缩维度操作将平均特征的冗余维度去除
        avg_feats = torch.squeeze(avg_feats, dim=2).squeeze(dim=2)
        # 图像特征通过线性层进行变换
        z_img = self.linear_z(avg_feats)
        # 使用线性层对图像特征进行分类，得到视觉标签
        vis_labels = self.classifier(z_img)
        # 获取记忆和掩码
        memory = self.get_mem(self.memory.to(images)).expand(bsz, -1, -1)
        mask = self.mask.to(images).expand(bsz, -1, -1)

        # 载入region文件
        # get images_id,检索对应特征文件 #
        global merged_tensor
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        directory = 'E:\Datasets\Regions\iu_region'  # 指定目录的路径
        if self.args.dataset_name == 'iu_xray':
            extension = '.npz'  # 文件的格式
            # 获取指定目录下的所有文件名
            files = os.listdir(directory)
            # 创建一个字典来存储每个前缀对应的文件列表
            file_dict = {}
            # 创建两个空列表，用于存储所有的 tensor
            all_tensors_0 = []
            # all_tensors_1 = []
            # 遍历文件列表
            for file in files:
                for target in image_id:
                    if file.startswith(target) and file.endswith(extension):
                        # 文件名以目标前缀开头且以指定的格式结尾，将文件添加到字典中
                        if target not in file_dict:
                            file_dict[target] = []
                        file_dict[target].append(file)
            # 遍历字典，提取特征
            for target, file_list in file_dict.items():
                if len(file_list) >= 2:
                    file_path0 = os.path.join(directory, file_list[0])
                    # file_path1 = os.path.join(directory, file_list[1])
                    # 在这里执行你的处理逻辑，使用 file_path1 和 file_path2 来访问文件

                    # 加载npz文件
                    data0 = np.load(file_path0, encoding='bytes', allow_pickle=True)
                    x0 = data0['x']
                    n0 = min(x0.shape[0], 10)  # 限制 n 不超过 10
                    region_feats_0 = torch.tensor(x0[:n0])  # 只取前 n 行
                    # region_feats_0 = torch.tensor(x0)
                    # data1 = np.load(file_path1, encoding='bytes', allow_pickle=True)
                    # x1 = data1['x']
                    # n1 = min(x0.shape[0], 10)  # 限制 n 不超过 10
                    # region_feats_1 = torch.tensor(x1[:n1])  # 只取前 n 行
                    # region_feats_1 = torch.tensor(x1)

                    # 将 tensor 添加到列表中
                    all_tensors_0.append(region_feats_0)
                    # all_tensors_1.append(region_feats_1)

                # 使用 torch.cat 函数将列表中的 tensor 沿着指定的维度拼接
                merged_tensor_0 = torch.stack(all_tensors_0, dim=0)
                # merged_tensor_1 = torch.stack(all_tensors_1, dim=0)
                # merged_tensor = torch.cat((merged_tensor_0, merged_tensor_1), dim=1)
                # 将 merged_tensor 移动到 GPU
                reg_tensor = merged_tensor_0.to(device)  # [16, 10, 2048]
        else: # 其他数据集，怎么把数据传入,我们只要reg_tensor这一个参数就够了
            reg_tensor = torch.zeros((images.shape[0], self.num_box, 2048)).to(device)
        if mode == 'train':
            # 如果是训练模式，则对目标序列进行编码，得到文本特征 txt_feats 和文本标签 txt_labels，并更新记忆
            txt_feats, z_txt, txt_labels = self.txt_encoder(targets)
            memory = self.update_memory(memory, txt_feats, mask)
        # 将视觉标签转换为嵌入向量
        emb_labels = self.embed_labels(vis_labels.unsqueeze(-1))
        # 使用选择先验模块选择注意力特征
        prior = self.select_prior(emb_labels, memory)
        att_feats = torch.cat((att_feats, self.linear_feat(prior)), dim=1)

        if mode == 'train':
            # 如果是训练模式，则调用编码器-解码器模块进行前向传播，并返回输出、视觉标签、文本标签、图像特征和文本特征
            output = self.encoder_decoder(att_feats, reg_tensor, targets, mode='forward')
            return (output, vis_labels, txt_labels, z_img, z_txt)
        elif mode == 'sample':
            # 如果是采样模式，则调用编码器-解码器模块进行采样，并返回输出和视觉标签
            output, _ = self.encoder_decoder(att_feats, reg_tensor, opt=self.args, mode='sample')
            return output, vis_labels
        else:
            raise ValueError


class LAMRGModel_v91(LAMRGModel_v12):
    """Ablation Study
        只用label的模型
    """

    def forward(self, images, targets=None, labels=None, mode='train'):
        bsz = images.shape[0]

        ve = self.forward_iu_xray if self.args.dataset_name == 'iu_xray' else self.forward_mimic_cxr
        att_feats, avg_feats, _ = ve(images)

        z_img = self.linear_z(avg_feats)
        vis_labels = self.classifier(z_img)

        txt_feats, z_txt, txt_labels = None, None, None
        # memory = self.get_mem(self.memory.to(images)).expand(bsz, -1, -1)
        # mask = self.mask.to(images).expand(bsz, -1, -1)
        # if mode == 'train':
        # txt_feats, z_txt, txt_labels = self.txt_encoder(targets)
        # memory = self.update_memory(memory, txt_feats, mask)

        # emb_labels = self.embed_labels(vis_labels.unsqueeze(-1))
        # prior = self.select_prior(emb_labels, memory)
        # att_feats = torch.cat((att_feats, self.linear_feat(prior)), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(avg_feats, att_feats, targets, mode='forward')
            return (output, vis_labels, txt_labels, z_img, z_txt)
        elif mode == 'sample':
            output, _ = self.encoder_decoder(avg_feats, att_feats, opt=self.args, mode='sample')
            return output, vis_labels
        else:
            raise ValueError


class LAMRGModel_v92(LAMRGModel_v12):
    """Ablation Study
        用label loss + rank loss的模型
    """

    def forward(self, images, targets=None, labels=None, mode='train'):
        bsz = images.shape[0]

        ve = self.forward_iu_xray if self.args.dataset_name == 'iu_xray' else self.forward_mimic_cxr
        att_feats, avg_feats, _ = ve(images)

        z_img = self.linear_z(avg_feats)
        vis_labels = self.classifier(z_img)

        # memory = self.get_mem(self.memory.to(images)).expand(bsz, -1, -1)
        # mask = self.mask.to(images).expand(bsz, -1, -1)
        if mode == 'train':
            txt_feats, z_txt, txt_labels = self.txt_encoder(targets)
            # memory = self.update_memory(memory, txt_feats, mask)

        # emb_labels = self.embed_labels(vis_labels.unsqueeze(-1))
        # prior = self.select_prior(emb_labels, memory)
        # att_feats = torch.cat((att_feats, self.linear_feat(prior)), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(avg_feats, att_feats, targets, mode='forward')
            return (output, vis_labels, txt_labels, z_img, z_txt)
        elif mode == 'sample':
            # ipdb.set_trace()
            output, _ = self.encoder_decoder(avg_feats, att_feats, opt=self.args, mode='sample')
            return output, vis_labels
        else:
            raise ValueError


class LAMRGModel_vRebuttal(LAMRGModel_v9):
    def __init__(self, args, tokenizer):
        super(LAMRGModel_vRebuttal, self).__init__(args, tokenizer)
        self.args = args
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_layers = args.num_layers
        self.num_labels = args.num_labels
        self.tgt_vocab = len(tokenizer.idx2token) + 1
        self.h = args.num_heads
        self.num_slots = args.num_slots
        self.d_vf = args.d_vf
        self.dropout = args.dropout

        self.txt_encoder = TextEncoder(self.d_model, self.d_ff, self.num_layers, self.tgt_vocab, self.num_labels,
                                       self.h, self.dropout)

        self.memory, self.mask = self.init_memory()

        self.get_mem = nn.Linear(self.d_model, self.d_model)
        self.linear_z = nn.Linear(self.d_vf, self.d_model)
        self.linear_feat = nn.Linear(self.d_model, self.d_vf)

        self.classifier = nn.Linear(self.d_model, self.num_labels)
        self.embed_labels = nn.Linear(1, self.d_model)

        self.init_weight()

    @staticmethod
    def attention(query, key, value):
        "Compute 'Dot Product Attention'"
        import math

        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1))
        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value)

    def init_weight(self):
        self._init_weight(self.linear_z)
        self._init_weight(self.get_mem)
        self._init_weight(self.linear_feat)
        self._init_weight(self.classifier)
        self._init_weight(self.embed_labels)

    def init_memory(self):
        memory = nn.Parameter(torch.eye(self.num_slots).unsqueeze(0))
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((1, self.num_slots, diff))
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]
        mask = torch.ones((self.num_slots, self.d_model))
        mask[:, self.num_slots:] = 0
        return memory, mask

    def forward(self, images, targets=None, labels=None, mode='train'):
        bsz = images.shape[0]

        ve = self.forward_iu_xray if self.args.dataset_name == 'iu_xray' else self.forward_mimic_cxr
        att_feats, avg_feats, _ = ve(images)

        z_img = self.linear_z(avg_feats)
        vis_labels = self.classifier(z_img)
        # ipdb.set_trace()
        memory = self.get_mem(self.memory.to(images)).expand(bsz, -1, -1)
        if mode == 'train':
            txt_feats, z_txt, txt_labels = self.txt_encoder(targets)
            memory = self.attention(memory, txt_feats, txt_feats)

        emb_labels = self.embed_labels(vis_labels.unsqueeze(-1))
        prior = self.attention(emb_labels, memory, memory)
        att_feats = torch.cat((att_feats, self.linear_feat(prior)), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(avg_feats, att_feats, targets, mode='forward')
            return (output, vis_labels, txt_labels, z_img, z_txt)
        elif mode == 'sample':
            output, _ = self.encoder_decoder(avg_feats, att_feats, opt=self.args, mode='sample')
            return output, vis_labels
        else:
            raise ValueError

class LAMRGModel_v98(LAMRGModel_v9):
    def __init__(self, args, tokenizer):
        super(LAMRGModel_v98, self).__init__(args, tokenizer)
        self.args = args
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_layers = args.num_layers
        self.num_labels = args.num_labels
        self.tgt_vocab = len(tokenizer.idx2token) + 1
        self.h = args.num_heads
        self.num_slots = args.num_slots
        self.d_vf = args.d_vf
        self.dropout = args.dropout
        self.num_box = args.num_box
        self.encoder_decoder_single = BaseCMN(args, tokenizer)
        # 初始化各个模块和网络层
        self.txt_encoder = TextEncoder(self.d_model, self.d_ff, self.num_layers, self.tgt_vocab, self.num_labels,
                                       self.h, self.dropout)
        self.update_memory = MHA_FF(self.d_model, self.d_ff, args.num_memory_heads, self.dropout)
        self.select_prior = MHA_FF(self.d_model, self.d_ff, args.num_memory_heads, self.dropout)
        self.memory, self.mask = self.init_memory()

        self.get_mem = nn.Linear(self.d_model, self.d_model)
        self.linear_z = nn.Linear(self.d_vf, self.d_model)
        self.linear_feat = nn.Linear(self.d_model, self.d_vf)

        self.classifier = nn.Linear(self.d_model, self.num_labels)
        self.embed_labels = nn.Linear(1, self.d_model)

        self.init_weight()

    def init_weight(self):
        # 初始化网络层的权重
        self._init_weight(self.linear_z)
        self._init_weight(self.get_mem)
        self._init_weight(self.linear_feat)
        self._init_weight(self.classifier)
        self._init_weight(self.embed_labels)

    def init_memory(self):
        # 初始化记忆矩阵和掩码
        memory = nn.Parameter(torch.eye(self.num_slots).unsqueeze(0))
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((1, self.num_slots, diff))
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]
        mask = torch.ones((self.num_slots, self.d_model))
        mask[:, self.num_slots:] = 0
        return memory, mask

    def forward_iu_xray(self, images):
        att_feats_0, fc_feats_0, labels_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1, labels_1 = self.visual_extractor(images[:, 1])

        fc_feats = torch.mean(torch.stack([fc_feats_0, fc_feats_1]), dim=0)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        out_labels = torch.mean(torch.stack([labels_0, labels_1]), dim=0)
        return att_feats, fc_feats, out_labels


    def forward(self, images_id, images, targets=None, labels=None, mode='train', update_opts={}):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        att_feats_0, avg_feats_0, _ = self.visual_extractor(images[:, 0])
        att_feats_1, avg_feats_1, _ = self.visual_extractor(images[:, 1])
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        # print(avg_feats.size())
        z_img = self.linear_z(avg_feats_0)
        reg_tensor = torch.zeros((images.shape[0], self.num_box, 2048)).to(device)
        vis_labels = self.classifier(z_img)
        if mode == 'train':
            txt_feats, z_txt, txt_labels = self.txt_encoder(targets)
            output = self.encoder_decoder_single(att_feats, reg_tensor, targets, mode='forward')
            return (output, vis_labels, txt_labels, z_img, z_txt)
            # return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder_single(att_feats, reg_tensor,  opt=self.args, mode='sample')
            return output, output_probs
        else:
            raise ValueError

class LAMRGModel_v99(LAMRGModel_v9):
    def __init__(self, args, tokenizer):
        super(LAMRGModel_v99, self).__init__(args, tokenizer)
        self.args = args
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_layers = args.num_layers
        self.num_labels = args.num_labels
        self.tgt_vocab = len(tokenizer.idx2token) + 1
        self.h = args.num_heads
        self.num_slots = args.num_slots
        self.d_vf = args.d_vf
        self.dropout = args.dropout
        self.num_box = args.num_box
        self.encoder_decoder_single = BaseCMN(args, tokenizer)
        # 初始化各个模块和网络层
        self.txt_encoder = TextEncoder(self.d_model, self.d_ff, self.num_layers, self.tgt_vocab, self.num_labels,
                                       self.h, self.dropout)
        self.update_memory = MHA_FF(self.d_model, self.d_ff, args.num_memory_heads, self.dropout)
        self.select_prior = MHA_FF(self.d_model, self.d_ff, args.num_memory_heads, self.dropout)
        self.memory, self.mask = self.init_memory()

        self.get_mem = nn.Linear(self.d_model, self.d_model)
        self.linear_z = nn.Linear(self.d_vf, self.d_model)
        self.linear_feat = nn.Linear(self.d_model, self.d_vf)

        self.classifier = nn.Linear(self.d_model, self.num_labels)
        self.embed_labels = nn.Linear(1, self.d_model)

        self.init_weight()

    def init_weight(self):
        # 初始化网络层的权重
        self._init_weight(self.linear_z)
        self._init_weight(self.get_mem)
        self._init_weight(self.linear_feat)
        self._init_weight(self.classifier)
        self._init_weight(self.embed_labels)

    def init_memory(self):
        # 初始化记忆矩阵和掩码
        memory = nn.Parameter(torch.eye(self.num_slots).unsqueeze(0))
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((1, self.num_slots, diff))
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]
        mask = torch.ones((self.num_slots, self.d_model))
        mask[:, self.num_slots:] = 0
        return memory, mask

    def forward(self, images_id, images, targets=None, labels=None, mode='train', update_opts={}):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        att_feats, avg_feats, out_labels = self.visual_extractor(images)
        # print(avg_feats.size())
        z_img = self.linear_z(avg_feats)
        reg_tensor = torch.zeros((images.shape[0], self.num_box, 2048)).to(device)
        vis_labels = self.classifier(z_img)
        if mode == 'train':
            txt_feats, z_txt, txt_labels = self.txt_encoder(targets)
            output = self.encoder_decoder_single(att_feats, reg_tensor, targets, mode='forward')
            return (output, vis_labels, txt_labels, z_img, z_txt)
            # return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder_single(att_feats, reg_tensor, mode='sample', opt=self.args)
            return output, output_probs
        else:
            raise ValueError


        # ve = self.forward_iu_xray if self.args.dataset_name == 'iu_xray' else self.forward_mimic_cxr
        # def forward_iu_xray(self, images):
        #     att_feats_0, fc_feats_0, labels_0 = self.visual_extractor(images[:, 0])
        #     att_feats_1, fc_feats_1, labels_1 = self.visual_extractor(images[:, 1])
        #
        #     fc_feats = torch.mean(torch.stack([fc_feats_0, fc_feats_1]), dim=0)
        #     att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        #
        #     out_labels = torch.mean(torch.stack([labels_0, labels_1]), dim=0)
        #     return att_feats, fc_feats, out_labels
        #
        # def forward_mimic_cxr(self, images):
        #     att_feats, fc_feats, out_labels = self.visual_extractor(images)
        #     return att_feats, fc_feats, out_labels