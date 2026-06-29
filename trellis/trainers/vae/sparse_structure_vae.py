from typing import *
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

from ..basic import BasicTrainer


class SparseStructureVaeTrainer(BasicTrainer):
    """
    稀疏结构变分自编码器（Sparse Structure VAE）训练器。
    
    Args:
        models (dict[str, nn.Module]): 待训练的模型。
        dataset (torch.utils.data.Dataset): 数据集。
        output_dir (str): 输出目录。
        load_dir (str): 加载目录。
        step (int): 要加载的训练步数。
        batch_size (int): 批大小（Batch Size）。
        batch_size_per_gpu (int): 单个 GPU 的批大小。如果指定此参数，batch_size 将被忽略。
        batch_split (int): 通过梯度累积进行批次拆分的份数。
        max_steps (int): 最大训练步数。
        optimizer (dict): 优化器配置。
        lr_scheduler (dict): 学习率调度器配置。
        elastic (dict): 弹性显存管理配置。
        grad_clip (float or dict): 梯度裁剪配置。
        ema_rate (float or list): 指数移动平均（EMA）的衰减率。
        fp16_mode (str): FP16 精度模式。
            - None: 不使用 FP16。
            - 'inflat_all': 为所有参数保留一个放大的 fp32 主参数（master parameter）。
            - 'amp': 自动混合精度（Automatic Mixed Precision）。
        fp16_scale_growth (float): FP16 梯度反向传播的缩放增长率。
        finetune_ckpt (dict): 微调检查点（Checkpoint）配置。
        log_param_stats (bool): 是否记录参数统计信息。
        i_print (int): 打印日志的步数间隔。
        i_log (int): 记录日志的步数间隔。
        i_sample (int): 采样的步数间隔。
        i_save (int): 保存检查点的步数间隔。
        i_ddpcheck (int): DDP（分布式数据并行）检查的步数间隔。
        
        loss_type (str): 损失类型。'bce' 表示二元交叉熵损失，'l1' 表示 L1 损失，'dice' 表示 Dice 损失。
        lambda_kl (float): KL 散度损失项的权重。
    """
    
    def __init__(
        self,
        *args,
        loss_type='bce',
        lambda_kl=1e-6,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.lambda_kl = lambda_kl

    # BEGIN detailed loss export: add weighted contribution fields for loss.txt.
    def export_loss_for_file(self, loss):
        export = super().export_loss_for_file(loss)
        contribution = {}
        for key in ('bce', 'l1', 'dice'):
            if key in export:
                contribution[key] = export[key]
        if 'kl' in export:
            contribution['kl'] = self.lambda_kl * export['kl']
        export['contribution'] = contribution
        return export
    # END detailed loss export.
    
    def training_losses(
        self,
        ss: torch.Tensor,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        计算训练损失。

        Args:
            ss: 二元稀疏结构张量，Shape 为 [N x 1 x H x W x D]。

        Returns:
            一个字典，其中键为 "loss" 的值包含一个标量损失张量。
            也可能包含对应其他损失项的其他键。
        """
        z, mean, logvar = self.training_models['encoder'](ss.float(), sample_posterior=True, return_raw=True)
        logits = self.training_models['decoder'](z)

        terms = edict(loss = 0.0)
        if self.loss_type == 'bce':
            terms["bce"] = F.binary_cross_entropy_with_logits(logits, ss.float(), reduction='mean')
            terms["loss"] = terms["loss"] + terms["bce"]
        elif self.loss_type == 'l1':
            terms["l1"] = F.l1_loss(F.sigmoid(logits), ss.float(), reduction='mean')
            terms["loss"] = terms["loss"] + terms["l1"]
        elif self.loss_type == 'dice':
            logits = F.sigmoid(logits)
            terms["dice"] = 1 - (2 * (logits * ss.float()).sum() + 1) / (logits.sum() + ss.float().sum() + 1)
            terms["loss"] = terms["loss"] + terms["dice"]
        else:
            raise ValueError(f'Invalid loss type {self.loss_type}')
        terms["kl"] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
        terms["loss"] = terms["loss"] + self.lambda_kl * terms["kl"]
            
        return terms, {}
    
    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=64, batch_size=1, verbose=False):
        super().snapshot(suffix=suffix, num_samples=num_samples, batch_size=batch_size, verbose=verbose)
    
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        """
        运行模型推理以生成快照（Snapshot）样本。

        使用训练集的数据通过 VAE 编码器和解码器，生成重建样本用于可视化评估。

        Args:
            num_samples (int): 要生成的快照样本总数。
            batch_size (int): 每个 GPU 的批大小。
            verbose (bool, optional): 是否打印详细的进度信息。默认为 False。

        Returns:
            Dict: 包含真值和重建结果的样本字典。结构如下：
                - 'gt' (dict):
                    - 'value' (torch.Tensor): 真实稀疏结构张量，Shape 为 [num_samples x 1 x H x W x D]。
                    - 'type' (str): 样本类型，固定为 'sample'。
                - 'recon' (dict):
                    - 'value' (torch.Tensor): 解码重建的二值稀疏结构张量，Shape 为 [num_samples x 1 x H x W x D]。
                    - 'type' (str): 样本类型，固定为 'sample'。
        """
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        # inference
        gts = []
        recons = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            args = {k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch] for k, v in data.items()}
            z = self.models['encoder'](args['ss'].float(), sample_posterior=False)
            logits = self.models['decoder'](z)
            recon = (logits > 0).long()
            gts.append(args['ss'])
            recons.append(recon)

        sample_dict = {
            'gt': {'value': torch.cat(gts, dim=0), 'type': 'sample'},
            'recon': {'value': torch.cat(recons, dim=0), 'type': 'sample'},
        }
        return sample_dict
