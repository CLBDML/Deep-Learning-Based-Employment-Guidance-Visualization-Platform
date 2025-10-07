"""
PyTorch + Transformer 模型实现（文本分类 & 语义匹配）
====================================================
仅聚焦深度学习网络与可训练脚手架，便于后续无缝对接后端服务（FastAPI / Flask / Tornado）
或 Spring Boot 网关。本文件包含：

1) 通用 Transformer 文本编码器（Embedding + Positional Embedding + nn.TransformerEncoder）
2) 文本分类模型（单句分类）
3) 语义匹配模型（两种范式）
   3.1 Bi-Encoder（双塔共享编码，支持 InfoNCE 对比学习 / 余弦相似度）
   3.2 Cross-Encoder（交叉编码器，拼接对输入，基于 [CLS] 表示分类）
4) 训练与评估脚手架（可选 HuggingFace Tokenizer；或使用外部已分词 id 序列）

依赖：
  - torch>=2.0
  - 可选：transformers>=4.0（用于Tokenizer，仅在有需要且环境允许时）

使用方式：
  A. 若已有 tokenizer 将文本转为 (input_ids, attention_mask)，可直接构建 Dataset & DataLoader。
  B. 若使用 HuggingFace tokenizer，将 `use_hf_tokenizer=True` 并提供 `pretrained_name_or_path`。

作者: ChatGPT (Hanbo 项目专用实现)
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ===============
# 基础模块
# ===============
class PositionalEncoding(nn.Module):
    """标准正弦位置编码（可学习位置嵌入可切换）。"""
    def __init__(self, d_model: int, max_len: int = 512, learnable: bool = False):
        super().__init__()
        self.d_model = d_model
        if learnable:
            self.pe = nn.Embedding(max_len, d_model)
            nn.init.normal_(self.pe.weight, mean=0.0, std=0.02)
            self.register_buffer("position_ids", torch.arange(max_len).long(), persistent=False)
            self.learnable = True
        else:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # [1, max_len, d_model]
            self.register_buffer('pe', pe, persistent=False)
            self.learnable = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        if self.learnable:
            L = x.size(1)
            pos_ids = self.position_ids[:L].unsqueeze(0).to(x.device)
            return x + self.pe(pos_ids)
        else:
            return x + self.pe[:, :x.size(1)]


class TransformerTextEncoder(nn.Module):
    """通用文本编码器：Embedding + PosEncoding + TransformerEncoder.

    参数
    ----
    vocab_size: 词表大小（若外部提供已映射 id，设置为足够大）
    d_model: 模型维度
    nhead: 注意力头数
    num_layers: TransformerEncoder 层数
    dim_feedforward: FFN 中间维度
    dropout: 丢弃率
    max_len: 最长序列
    use_cls_token: 是否使用专门的 [CLS] token 作为池化
    pad_id: padding id
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 512,
        use_cls_token: bool = True,
        pad_id: int = 0,
        learnable_pos: bool = False,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.use_cls_token = use_cls_token
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)

        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, learnable=learnable_pos)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if self.use_cls_token:
            # 预留一个 [CLS] 的 embedding（可选择词表中固定 id，如1），这里直接参数化一个向量
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        输入：
          input_ids: [B, L]
          attention_mask: [B, L]，1 表示有效，0 表示 padding
        输出：
          seq_out: [B, L'] 编码后的序列特征（若 use_cls_token=True，则 L' = L+1）
        """
        B, L = input_ids.shape
        x = self.tok_emb(input_ids)  # [B, L, D]
        if self.use_cls_token:
            cls_tok = self.cls.expand(B, -1, -1)  # [B, 1, D]
            x = torch.cat([cls_tok, x], dim=1)    # [B, L+1, D]
            if attention_mask is not None:
                attention_mask = torch.cat([
                    torch.ones(B, 1, device=input_ids.device, dtype=attention_mask.dtype),
                    attention_mask
                ], dim=1)
        x = self.pos_enc(x)

        if attention_mask is None:
            src_key_padding_mask = (input_ids == self.pad_id)
            if self.use_cls_token:
                pad_col = torch.zeros(B, 1, device=input_ids.device, dtype=torch.bool)
                src_key_padding_mask = torch.cat([pad_col, src_key_padding_mask], dim=1)
        else:
            src_key_padding_mask = attention_mask == 0  # True for pad

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x  # [B, L' , D]

    def pool(self, seq_out: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, method: str = 'cls') -> torch.Tensor:
        """将序列特征池化到一个向量。
        method: 'cls' | 'mean' | 'max'
        """
        if method == 'cls' and self.use_cls_token:
            return seq_out[:, 0]  # [B, D]
        elif method == 'mean':
            if attention_mask is not None:
                if self.use_cls_token:
                    attention_mask = torch.cat([
                        torch.ones(attention_mask.size(0), 1, device=attention_mask.device, dtype=attention_mask.dtype),
                        attention_mask
                    ], dim=1)
                mask = attention_mask.unsqueeze(-1)  # [B, L, 1]
                summed = (seq_out * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1e-6)
                return summed / denom
            else:
                return seq_out.mean(dim=1)
        elif method == 'max':
            if attention_mask is not None:
                if self.use_cls_token:
                    attention_mask = torch.cat([
                        torch.ones(attention_mask.size(0), 1, device=attention_mask.device, dtype=attention_mask.dtype),
                        attention_mask
                    ], dim=1)
                masked = seq_out.masked_fill((attention_mask == 0).unsqueeze(-1), float('-inf'))
                return masked.max(dim=1).values
            else:
                return seq_out.max(dim=1).values
        else:
            # 回退到 cls
            return seq_out[:, 0]


# ===============
# 文本分类
# ===============
class TextTransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int, **encoder_kwargs):
        super().__init__()
        self.encoder = TransformerTextEncoder(vocab_size, **encoder_kwargs)
        d_model = encoder_kwargs.get('d_model', 256)
        self.dropout = nn.Dropout(encoder_kwargs.get('dropout', 0.1))
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        seq = self.encoder(input_ids, attention_mask)
        pooled = self.encoder.pool(seq, attention_mask, method='cls')
        logits = self.classifier(self.dropout(pooled))
        out = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            out["loss"] = loss
        return out


# ===============
# 语义匹配 - Bi-Encoder (对比学习 / 向量相似)
# ===============
class TextTransformerBiEncoder(nn.Module):
    def __init__(self, vocab_size: int, proj_dim: Optional[int] = None, temperature: float = 0.05, **encoder_kwargs):
        super().__init__()
        self.encoder = TransformerTextEncoder(vocab_size, **encoder_kwargs)
        d_model = encoder_kwargs.get('d_model', 256)
        self.proj = nn.Linear(d_model, proj_dim) if proj_dim else nn.Identity()
        self.temperature = temperature

    def encode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, pool: str = 'cls') -> torch.Tensor:
        seq = self.encoder(input_ids, attention_mask)
        pooled = self.encoder.pool(seq, attention_mask, method=pool)
        z = self.proj(pooled)
        z = F.normalize(z, dim=-1)
        return z  # [B, D]

    def forward(self,
                a_input_ids: torch.Tensor,
                a_attention_mask: Optional[torch.Tensor],
                b_input_ids: torch.Tensor,
                b_attention_mask: Optional[torch.Tensor],
                labels: Optional[torch.Tensor] = None,
                in_batch_negatives: bool = True):
        """
        若 labels=None 且 in_batch_negatives=True：采用 InfoNCE (对齐同索引样本)，默认 batch 内一一对应。
        若 labels 提供为相似度矩阵的行索引（或 one-hot），则支持自定义正样本。
        返回：相似度矩阵 sim，及可选 loss。
        """
        za = self.encode(a_input_ids, a_attention_mask)
        zb = self.encode(b_input_ids, b_attention_mask)
        sim = za @ zb.t()  # [B, B]
        sim = sim / self.temperature
        out = {"similarity": sim, "za": za, "zb": zb}

        if labels is None and in_batch_negatives:
            target = torch.arange(sim.size(0), device=sim.device)
            loss = F.cross_entropy(sim, target)
            out["loss"] = loss
        elif labels is not None:
            # 若 labels 是 shape [B] 的目标索引
            if labels.dim() == 1:
                out["loss"] = F.cross_entropy(sim, labels)
            else:
                # 若 labels 是 one-hot 或 soft 标签 [B, B]
                log_prob = F.log_softmax(sim, dim=-1)
                out["loss"] = -(labels * log_prob).sum(dim=-1).mean()
        return out


# ===============
# 语义匹配 - Cross-Encoder（拼接输入做分类）
# ===============
class CrossEncoderForMatching(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int = 2, sep_id: Optional[int] = None, **encoder_kwargs):
        """
        将句对拼接为： [CLS] A ... [SEP] B ...
        注意：需要在外部数据处理时做好拼接（或使用本类的 collate_fn_v2 组装）。
        """
        super().__init__()
        self.encoder = TransformerTextEncoder(vocab_size, **encoder_kwargs)
        d_model = encoder_kwargs.get('d_model', 256)
        self.dropout = nn.Dropout(encoder_kwargs.get('dropout', 0.1))
        self.classifier = nn.Linear(d_model, num_labels)
        self.sep_id = sep_id

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        seq = self.encoder(input_ids, attention_mask)
        pooled = self.encoder.pool(seq, attention_mask, method='cls')
        logits = self.classifier(self.dropout(pooled))
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(logits, labels)
        return out


# ===============
# 数据集示例（两种：单句分类；句对匹配）
# ===============
class SimpleTextClsDataset(Dataset):
    """假设已经完成分词 -> id 映射，传入 input_ids 与 attention_mask。"""
    def __init__(self, encodings: Dict[str, List[List[int]]], labels: List[int], pad_id: int = 0, max_len: int = 256):
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings.get('attention_mask', None)
        self.labels = labels
        self.pad_id = pad_id
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def _pad(self, ids: List[int]) -> Tuple[List[int], List[int]]:
        ids = ids[: self.max_len]
        attn = [1] * len(ids)
        if len(ids) < self.max_len:
            pad_len = self.max_len - len(ids)
            ids = ids + [self.pad_id] * pad_len
            attn = attn + [0] * pad_len
        return ids, attn

    def __getitem__(self, idx):
        ids = self.input_ids[idx]
        ids, attn = self._pad(ids)
        y = self.labels[idx]
        return {
            'input_ids': torch.tensor(ids).long(),
            'attention_mask': torch.tensor(attn).long(),
            'labels': torch.tensor(y).long(),
        }


class SimplePairMatchDataset(Dataset):
    """句对匹配数据集：A、B 两侧各自独立编码，用于 Bi-Encoder。"""
    def __init__(self, enc_a: Dict[str, List[List[int]]], enc_b: Dict[str, List[List[int]]], labels: Optional[List[int]] = None,
                 pad_id: int = 0, max_len: int = 256):
        assert len(enc_a['input_ids']) == len(enc_b['input_ids'])
        self.a_ids = enc_a['input_ids']
        self.a_mask = enc_a.get('attention_mask', None)
        self.b_ids = enc_b['input_ids']
        self.b_mask = enc_b.get('attention_mask', None)
        self.labels = labels
        self.pad_id = pad_id
        self.max_len = max_len

    def __len__(self):
        return len(self.a_ids)

    def _pad(self, ids: List[int]) -> Tuple[List[int], List[int]]:
        ids = ids[: self.max_len]
        attn = [1] * len(ids)
        if len(ids) < self.max_len:
            pad_len = self.max_len - len(ids)
            ids = ids + [self.pad_id] * pad_len
            attn = attn + [0] * pad_len
        return ids, attn

    def __getitem__(self, idx):
        a = self.a_ids[idx]
        b = self.b_ids[idx]
        a, am = self._pad(a)
        b, bm = self._pad(b)
        item = {
            'a_input_ids': torch.tensor(a).long(),
            'a_attention_mask': torch.tensor(am).long(),
            'b_input_ids': torch.tensor(b).long(),
            'b_attention_mask': torch.tensor(bm).long(),
        }
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx]).long()
        return item


# ===============
# 训练器（简化版）
# ===============
@dataclass
class TrainConfig:
    lr: float = 5e-4
    batch_size: int = 32
    epochs: int = 5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


class ClassifierTrainer:
    def __init__(self, model: TextTransformerClassifier, cfg: TrainConfig):
        self.model = model.to(cfg.device)
        self.cfg = cfg
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.device.startswith('cuda')))

    def train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        losses, accs = [], []
        for batch in loader:
            input_ids = batch['input_ids'].to(self.cfg.device)
            attention_mask = batch['attention_mask'].to(self.cfg.device)
            labels = batch['labels'].to(self.cfg.device)
            self.opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(self.cfg.device.startswith('cuda'))):
                out = self.model(input_ids, attention_mask, labels)
            loss = out['loss']
            acc = accuracy_from_logits(out['logits'], labels)
            self.scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            losses.append(loss.item())
            accs.append(acc)
        return float(sum(losses)/len(losses)), float(sum(accs)/len(accs))

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        losses, accs = [], []
        for batch in loader:
            input_ids = batch['input_ids'].to(self.cfg.device)
            attention_mask = batch['attention_mask'].to(self.cfg.device)
            labels = batch['labels'].to(self.cfg.device)
            out = self.model(input_ids, attention_mask, labels)
            loss = out['loss']
            acc = accuracy_from_logits(out['logits'], labels)
            losses.append(loss.item())
            accs.append(acc)
        return float(sum(losses)/len(losses)), float(sum(accs)/len(accs))


class BiEncoderTrainer:
    def __init__(self, model: TextTransformerBiEncoder, cfg: TrainConfig):
        self.model = model.to(cfg.device)
        self.cfg = cfg
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.cfg.device.startswith('cuda')))

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        losses = []
        for batch in loader:
            a_ids = batch['a_input_ids'].to(self.cfg.device)
            a_mask = batch['a_attention_mask'].to(self.cfg.device)
            b_ids = batch['b_input_ids'].to(self.cfg.device)
            b_mask = batch['b_attention_mask'].to(self.cfg.device)
            labels = batch.get('labels', None)
            if labels is not None:
                labels = labels.to(self.cfg.device)
            self.opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(self.cfg.device.startswith('cuda'))):
                out = self.model(a_ids, a_mask, b_ids, b_mask, labels=labels, in_batch_negatives=(labels is None))
            loss = out['loss']
            self.scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            losses.append(loss.item())
        return float(sum(losses)/len(losses))

    @torch.no_grad()
    def evaluate_recall_at_k(self, loader: DataLoader, k: int = 10) -> float:
        """评估召回@k（以 batch 为单位近似，真实检索请离线构建库）。"""
        self.model.eval()
        correct, total = 0, 0
        for batch in loader:
            a_ids = batch['a_input_ids'].to(self.cfg.device)
            a_mask = batch['a_attention_mask'].to(self.cfg.device)
            b_ids = batch['b_input_ids'].to(self.cfg.device)
            b_mask = batch['b_attention_mask'].to(self.cfg.device)
            za = self.model.encode(a_ids, a_mask)
            zb = self.model.encode(b_ids, b_mask)
            sim = za @ zb.t()  # [B, B]
            topk = sim.topk(k, dim=-1).indices  # [B, k]
            target = torch.arange(sim.size(0), device=sim.device).unsqueeze(-1)
            correct += (topk == target).any(dim=-1).sum().item()
            total += sim.size(0)
        return correct / max(total, 1)


# ===============
# 示例：快速自测（随机数据）
# ===============
if __name__ == "__main__":
    torch.manual_seed(42)
    vocab_size = 32000
    max_len = 64

    # ---- 文本分类自测 ----
    num_labels = 5
    model_cls = TextTransformerClassifier(vocab_size=vocab_size, num_labels=num_labels,
                                          d_model=256, nhead=8, num_layers=4,
                                          dim_feedforward=768, dropout=0.1,
                                          max_len=max_len, use_cls_token=True, pad_id=0)

    # 构造随机数据
    N = 256
    X = [[random.randint(5, 500) for _ in range(random.randint(8, max_len))] for _ in range(N)]
    y = [random.randint(0, num_labels-1) for _ in range(N)]
    ds = SimpleTextClsDataset({'input_ids': X}, y, pad_id=0, max_len=max_len)
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    cfg = TrainConfig(epochs=2, batch_size=32, lr=3e-4)
    trainer = ClassifierTrainer(model_cls, cfg)
    for epoch in range(cfg.epochs):
        tr_loss, tr_acc = trainer.train_epoch(loader)
        print(f"[CLS-Classifier] Epoch {epoch+1} | loss={tr_loss:.4f} | acc={tr_acc:.4f}")

    # ---- 语义匹配（Bi-Encoder）自测 ----
    model_bi = TextTransformerBiEncoder(vocab_size=vocab_size, proj_dim=128, temperature=0.05,
                                        d_model=256, nhead=8, num_layers=4,
                                        dim_feedforward=768, dropout=0.1,
                                        max_len=max_len, use_cls_token=True, pad_id=0)

    Xa = [[random.randint(5, 500) for _ in range(random.randint(8, max_len))] for _ in range(N)]
    Xb = [x[:] for x in Xa]  # 正例构造：B 复制 A（仅用于自测）
    ds_pair = SimplePairMatchDataset({'input_ids': Xa}, {'input_ids': Xb}, labels=None, pad_id=0, max_len=max_len)
    loader_pair = DataLoader(ds_pair, batch_size=32, shuffle=True)

    bi_trainer = BiEncoderTrainer(model_bi, TrainConfig(epochs=2, batch_size=32, lr=3e-4))
    for epoch in range(2):
        loss = bi_trainer.train_epoch(loader_pair)
        r10 = bi_trainer.evaluate_recall_at_k(loader_pair, k=10)
        print(f"[Bi-Encoder] Epoch {epoch+1} | loss={loss:.4f} | R@10={r10:.4f}")

    # ---- Cross-Encoder 说明 ----
    # Cross-Encoder 训练通常需要在 collate 时把句对拼接到一起；
    # 这里不做冗长示例，实际生产建议：
    # input = concat([CLS], A, [SEP], B) -> 喂给 CrossEncoderForMatching 做二分类。
    # model_cross = CrossEncoderForMatching(vocab_size=vocab_size, num_labels=2, d_model=256, nhead=8, ...)
    # 训练过程与 TextTransformerClassifier 类似，labels 为 {0,1}。
