# -*- coding: utf-8 -*-
"""
Shenzhen Solitaire 求解器 v7.0 (PyTorch版)
主要改进：
1. 使用PyTorch实现神经网络
2. 更强大的网络架构（残差网络 + Attention）
3. 经验回放 + 优先级采样
4. 目标网络 + 软更新
5. 完善的checkpointing机制
"""

import heapq
import time
import hashlib
import random
import numpy as np
import pickle
import os
from typing import List, Optional, Tuple, Dict, Set
from enum import Enum
from collections import deque, namedtuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 设置设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# ==================== 基础数据结构 ====================

class CardType(Enum):
    NUMBER = 0
    FLOWER = 1
    SPECIAL = 2

class Card:
    __slots__ = ("value",)
    def __init__(self, value: int):
        self.value = value

    @property
    def type(self) -> CardType:
        if self.value < 27: return CardType.NUMBER
        if self.value < 39: return CardType.FLOWER
        return CardType.SPECIAL

    @property
    def color(self) -> int:
        if self.type == CardType.NUMBER:
            return self.value // 9
        if self.type == CardType.FLOWER:
            return (self.value - 27) // 4
        raise ValueError("Special card has no color")

    @property
    def num(self) -> int:
        if self.type == CardType.NUMBER:
            return self.value % 9 + 1
        return 0

    def __repr__(self):
        if self.type == CardType.NUMBER:
            return f"{self.num}{'RGB'[self.color]}"
        elif self.type == CardType.FLOWER:
            return f"F{self.color+1}"
        return "SP"

    def __eq__(self, other):
        return isinstance(other, Card) and self.value == other.value

    def __hash__(self):
        return hash(self.value)


class Stack:
    __slots__ = ("cards", "cont_len")
    
    def __init__(self, cards: Optional[List[Card]] = None):
        self.cards: List[Card] = list(cards) if cards else []
        self.cont_len = 0
        self._update_cont()

    def push(self, card: Card):
        self.cards.append(card)
        self._update_cont()

    def pop(self) -> Card:
        card = self.cards.pop()
        self.cont_len = max(0, self.cont_len - 1)
        if self.cont_len == 0 and self.cards:
            self._update_cont()
        return card

    def top(self) -> Optional[Card]:
        return self.cards[-1] if self.cards else None

    def movable_count(self) -> int:
        return self.cont_len

    def _update_cont(self):
        if not self.cards:
            self.cont_len = 0
            return
        length = 1
        for i in range(len(self.cards)-2, -1, -1):
            cur = self.cards[i+1]
            prev = self.cards[i]
            if (cur.type == CardType.NUMBER and prev.type == CardType.NUMBER and
                cur.color != prev.color and cur.num == prev.num - 1):
                length += 1
            else:
                break
        self.cont_len = length

    def __len__(self):
        return len(self.cards)


class Status:
    __slots__ = ("stacks", "stash", "stash_limit", "sorted_tops", "special_removed")
    
    def __init__(self):
        self.stacks: List[Stack] = [Stack() for _ in range(8)]
        self.stash: Set[Card] = set()
        self.stash_limit = 3
        self.sorted_tops = [0, 0, 0]
        self.special_removed = False

    def print_status(self, compact: bool = False):
        if compact:
            print(f"Stash:{len(self.stash)}/{self.stash_limit} | 已收:R{self.sorted_tops[0]}G{self.sorted_tops[1]}B{self.sorted_tops[2]}SP{int(self.special_removed)}")
        else:
            stash_cards = sorted(list(self.stash), key=lambda c: c.value)
            stash_str = " ".join(repr(c) for c in stash_cards) if stash_cards else "空"
            print(f"Stash({len(stash_cards)}/{self.stash_limit}): {stash_str}")
            print(f"已收: 红:{self.sorted_tops[0]} 绿:{self.sorted_tops[1]} 蓝:{self.sorted_tops[2]}  特殊牌:{['未收','已收'][self.special_removed]}")
            
            max_height = max((len(st.cards) for st in self.stacks), default=0)
            if max_height > 0:
                print("     " + "".join(f"列{i+1:2}        " for i in range(8)))
                for row in range(max_height):
                    line = f"{row+1:2}:  "
                    for st in self.stacks:
                        if row < len(st.cards):
                            card = st.cards[row]
                            marker = "←" if row >= len(st.cards) - st.cont_len else " "
                            line += f"{marker}{repr(card):10} "
                        else:
                            line += "            "
                    print(line)
            else:
                print("     (所有列为空)")
            print()

    def hash_key(self) -> bytes:
        stacks_repr = sorted([tuple(c.value for c in st.cards) for st in self.stacks])
        parts = []
        MAX_LEN = 20
        for col in stacks_repr:
            col_list = list(col)[:MAX_LEN]
            parts.extend(col_list + [255] * (MAX_LEN - len(col_list)))
        
        stash_vals = sorted(c.value for c in self.stash)
        parts.extend(stash_vals + [255] * (3 - len(stash_vals)))
        parts.extend(self.sorted_tops)
        parts.append(3 - self.stash_limit)
        parts.append(int(self.special_removed))
        
        return hashlib.md5(bytes(parts)).digest()

    def _can_auto_remove(self, card: Card) -> bool:
        if card.type == CardType.SPECIAL:
            return True
        if card.type != CardType.NUMBER:
            return False
        
        color, need = card.color, card.num
        top = self.sorted_tops[color]
        
        if top + 1 == need:
            if need <= 2:
                return True
            other_min = min(self.sorted_tops[(color+1)%3], self.sorted_tops[(color+2)%3])
            return other_min >= top - 1
        return False

    def _auto_remove_flowers(self) -> bool:
        count = [0, 0, 0]
        sources = []
        
        for i, st in enumerate(self.stacks):
            t = st.top()
            if t and t.type == CardType.FLOWER:
                count[t.color] += 1
                sources.append((True, i, t))
        
        for c in self.stash:
            if c.type == CardType.FLOWER:
                count[c.color] += 1
                sources.append((False, -1, c))
        
        for color in range(3):
            if count[color] == 4:
                has_slot = len(self.stash) < self.stash_limit
                has_same_in_stash = any(c.type == CardType.FLOWER and c.color == color for c in self.stash)
                
                if has_slot or has_same_in_stash:
                    removed = 0
                    for is_stack, idx, card in sources:
                        if removed == 4:
                            break
                        if card.type == CardType.FLOWER and card.color == color:
                            if is_stack:
                                if self.stacks[idx].top() and self.stacks[idx].top().value == card.value:
                                    self.stacks[idx].pop()
                                    removed += 1
                            else:
                                if card in self.stash:
                                    self.stash.remove(card)
                                    removed += 1
                    
                    self.stash_limit = max(0, self.stash_limit - 1)
                    return True
        
        return False

    def auto_remove(self):
        changed = True
        while changed:
            changed = False
            
            for st in self.stacks:
                t = st.top()
                if t and self._can_auto_remove(t):
                    card = st.pop()
                    if card.type == CardType.SPECIAL:
                        self.special_removed = True
                    else:
                        self.sorted_tops[card.color] = card.num
                    changed = True
                    break
            
            if changed:
                continue
            
            for c in list(self.stash):
                if self._can_auto_remove(c):
                    self.stash.remove(c)
                    if c.type == CardType.SPECIAL:
                        self.special_removed = True
                    else:
                        self.sorted_tops[c.color] = c.num
                    changed = True
                    break
            
            if changed:
                continue
            
            if self._auto_remove_flowers():
                changed = True

    def is_solved(self) -> bool:
        if not self.special_removed:
            return False
        for st in self.stacks:
            if st.cards:
                return False
        for c in self.stash:
            if c.type != CardType.FLOWER:
                return False
        return True

    def copy(self) -> 'Status':
        new_s = Status()
        for i, st in enumerate(self.stacks):
            new_s.stacks[i].cards = st.cards.copy()
            new_s.stacks[i].cont_len = st.cont_len
        new_s.stash = {Card(c.value) for c in self.stash}
        new_s.stash_limit = self.stash_limit
        new_s.sorted_tops = self.sorted_tops[:]
        new_s.special_removed = self.special_removed
        return new_s


# ==================== 增强特征提取器 ====================

class EnhancedFeatureExtractor:
    """提取更丰富的特征（扩展到128维）"""
    
    @staticmethod
    def extract(status: Status) -> np.ndarray:
        features = []
        
        # 每列详细信息 (8列 × 10特征 = 80维)
        for st in status.stacks:
            # 基础特征
            features.append(len(st.cards) / 5.0)
            features.append(st.cont_len / 5.0)
            
            # 顶牌信息
            top = st.top()
            if top and top.type == CardType.NUMBER:
                features.extend([top.color / 2.0, top.num / 9.0])
            else:
                features.extend([0.0, 0.0])
            
            features.append(1.0 if not st.cards else 0.0)
            features.append(sum(1 for c in st.cards if c.type == CardType.NUMBER) / 5.0)
            
            # 新增：阻塞度（有多少牌被压在下面）
            blocked = 0
            for i, card in enumerate(st.cards):
                if card.type == CardType.NUMBER:
                    if i < len(st.cards) - 1:  # 不是顶牌
                        blocked += 1
            features.append(blocked / 5.0)
            
            # 新增：列的"有序度"
            ordered = 0
            for i in range(len(st.cards) - 1):
                if (st.cards[i+1].type == CardType.NUMBER and 
                    st.cards[i].type == CardType.NUMBER and
                    st.cards[i+1].color != st.cards[i].color and
                    st.cards[i+1].num == st.cards[i].num - 1):
                    ordered += 1
            features.append(ordered / 4.0)
            
            # 新增：底牌信息
            if st.cards:
                bottom = st.cards[0]
                if bottom.type == CardType.NUMBER:
                    features.extend([bottom.color / 2.0, bottom.num / 9.0])
                else:
                    features.extend([0.0, 0.0])
            else:
                features.extend([0.0, 0.0])
        
        # Stash信息 (增强到12维)
        features.append(len(status.stash) / 3.0)
        features.append(status.stash_limit / 3.0)
        features.append(sum(1 for c in status.stash if c.type == CardType.NUMBER) / 3.0)
        features.append(sum(1 for c in status.stash if c.type == CardType.FLOWER) / 3.0)
        features.append(1.0 if len(status.stash) >= status.stash_limit else 0.0)
        
        # 每种颜色的花牌数量
        flower_count = [0, 0, 0]
        for c in status.stash:
            if c.type == CardType.FLOWER:
                flower_count[c.color] += 1
        features.extend([x / 4.0 for x in flower_count])
        
        # Stash中最小和最大数字牌
        num_cards_in_stash = [c.num for c in status.stash if c.type == CardType.NUMBER]
        if num_cards_in_stash:
            features.append(min(num_cards_in_stash) / 9.0)
            features.append(max(num_cards_in_stash) / 9.0)
        else:
            features.extend([0.0, 0.0])
        
        # Stash是否有关键牌（能打出的）
        has_playable = 0
        for c in status.stash:
            if c.type == CardType.NUMBER:
                for st in status.stacks:
                    top = st.top()
                    if not top or (top.type == CardType.NUMBER and 
                                  c.color != top.color and c.num == top.num - 1):
                        has_playable = 1
                        break
        features.append(float(has_playable))
        
        # 收集进度 (4维)
        features.extend([x / 9.0 for x in status.sorted_tops])
        features.append(1.0 if status.special_removed else 0.0)
        
        # 全局统计 (扩展到16维)
        total_cards = sum(len(st.cards) for st in status.stacks) + len(status.stash)
        features.append(total_cards / 40.0)
        features.append(sum(status.sorted_tops) / 27.0)
        
        empty_cols = sum(1 for st in status.stacks if not st.cards)
        features.append(empty_cols / 8.0)
        
        total_cont = sum(st.cont_len for st in status.stacks)
        features.append(total_cont / 40.0)
        
        progress = (sum(status.sorted_tops) + int(status.special_removed) * 3) / 30.0
        features.append(progress)
        
        # 新增：每种颜色的进度差异
        max_color = max(status.sorted_tops)
        min_color = min(status.sorted_tops)
        features.append((max_color - min_color) / 9.0)
        
        # 新增：场上可移动的牌组数
        movable_groups = sum(1 for st in status.stacks if st.cont_len > 0)
        features.append(movable_groups / 8.0)
        
        # 新增：下一步需要的牌的可达性
        for color in range(3):
            need = status.sorted_tops[color] + 1
            if need > 9:
                features.append(1.0)  # 已完成
            else:
                # 检查这张牌的深度
                min_depth = 99
                for st in status.stacks:
                    for d, c in enumerate(reversed(st.cards), 1):
                        if c.type == CardType.NUMBER and c.color == color and c.num == need:
                            min_depth = min(min_depth, d)
                            break
                for c in status.stash:
                    if c.type == CardType.NUMBER and c.color == color and c.num == need:
                        min_depth = 1
                features.append(1.0 - min(min_depth, 10) / 10.0)
        
        # 新增：死锁检测特征
        # 检查是否有牌永远无法打出
        deadlock_score = 0
        for st in status.stacks:
            if len(st.cards) > 1:
                # 检查底部的牌
                for i, card in enumerate(st.cards[:-1]):
                    if card.type == CardType.NUMBER:
                        # 这张牌需要的前驱
                        need_color = card.color
                        need_num = card.num + 1
                        # 检查前驱是否被压在它下面
                        for j in range(i):
                            below = st.cards[j]
                            if (below.type == CardType.NUMBER and 
                                below.color != need_color and below.num == need_num):
                                deadlock_score += 1
        features.append(deadlock_score / 10.0)
        
        # 填充到128维
        while len(features) < 128:
            features.append(0.0)
        
        return np.array(features[:128], dtype=np.float32)


# ==================== PyTorch价值网络（残差+Attention）====================

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += residual
        out = F.relu(out)
        return out


class AttentionModule(nn.Module):
    """简单的自注意力模块"""
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim // 4)
        self.key = nn.Linear(dim, dim // 4)
        self.value = nn.Linear(dim, dim)
        self.scale = (dim // 4) ** 0.5
        
    def forward(self, x):
        # x: (batch, dim)
        q = self.query(x).unsqueeze(1)  # (batch, 1, dim//4)
        k = self.key(x).unsqueeze(1)    # (batch, 1, dim//4)
        v = self.value(x).unsqueeze(1)  # (batch, 1, dim)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).squeeze(1)
        return out


class EnhancedValueNetwork(nn.Module):
    """增强的价值网络"""
    def __init__(self, input_size=128, hidden_size=256):
        super().__init__()
        
        # 输入层
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 残差块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(3)
        ])
        
        # 注意力模块
        self.attention = AttentionModule(hidden_size)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # 输入处理
        out = self.input_layer(x)
        
        # 残差块
        for res_block in self.res_blocks:
            out = res_block(out)
        
        # 注意力
        attn_out = self.attention(out)
        
        # 拼接
        combined = torch.cat([out, attn_out], dim=-1)
        
        # 输出
        value = self.output_layer(combined)
        return value


# ==================== 经验回放缓冲区 ====================

Experience = namedtuple('Experience', ['state', 'value', 'priority'])

class PrioritizedReplayBuffer:
    """优先级经验回放"""
    def __init__(self, capacity=50000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.buffer = []
        self.priorities = []
        self.position = 0
        
    def push(self, state, value, priority=None):
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(Experience(state, value, priority))
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = Experience(state, value, priority)
            self.priorities[self.position] = priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        """根据优先级采样"""
        if len(self.buffer) == 0:
            return [], [], []
        
        # 计算采样概率
        priorities = np.array(self.priorities[:len(self.buffer)])
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 采样
        indices = np.random.choice(len(self.buffer), 
                                  size=min(batch_size, len(self.buffer)),
                                  p=probs, replace=False)
        
        samples = [self.buffer[idx] for idx in indices]
        
        # 重要性采样权重
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)


# ==================== RL训练器（PyTorch版） ====================

class PyTorchRLTrainer:
    """使用PyTorch的RL训练器"""
    
    def __init__(self, learning_rate=0.0003, hidden_size=256):
        self.feature_extractor = EnhancedFeatureExtractor()
        self.device = DEVICE
        
        # 主网络
        self.value_net = EnhancedValueNetwork(128, hidden_size).to(self.device)
        # 目标网络
        self.target_net = EnhancedValueNetwork(128, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.value_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 经验回放
        self.replay_buffer = PrioritizedReplayBuffer(capacity=50000)
        
        # 训练统计
        self.train_step = 0
        self.target_update_freq = 500
        self.tau = 0.005  # 软更新系数
        
        self.best_loss = float('inf')
        self.checkpoint_dir = "checkpoints"
        self.expert_data_dir = "expert_data"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.expert_data_dir, exist_ok=True)
        
    def load_expert_data(self, filepath: str) -> bool:
        """从文件加载专家经验"""
        if not os.path.exists(filepath):
            print(f"✗ 专家数据文件不存在: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            experiences = data['experiences']
            metadata = data['metadata']
            
            print(f"✓ 成功加载专家数据: {filepath}")
            print(f"  采集日期: {metadata.get('collection_date', '未知')}")
            print(f"  经验数量: {len(experiences)}")
            print(f"  成功盘面: {metadata.get('successful', '?')}/{metadata.get('num_games', '?')}")
            
            # 添加到经验回放缓冲区
            for exp in experiences:
                self.replay_buffer.push(
                    exp['features'],
                    exp['value'],
                    exp.get('priority', 1.0)
                )
            
            print(f"  ✓ 已加载到经验池，当前大小: {len(self.replay_buffer)}")
            return True
            
        except Exception as e:
            print(f"✗ 加载失败: {e}")
            return False
    
    def list_expert_data_files(self) -> List[str]:
        """列出可用的专家数据文件"""
        if not os.path.exists(self.expert_data_dir):
            return []
        
        files = [f for f in os.listdir(self.expert_data_dir) if f.endswith('.pkl')]
        files.sort(reverse=True)  # 最新的在前
        return files
    
    def soft_update_target(self):
        """软更新目标网络"""
        for target_param, param in zip(self.target_net.parameters(), 
                                       self.value_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
    
    def train_from_loaded_data(self, epochs=10, batch_size=64, verbose=True) -> bool:
        """从已加载的经验数据训练网络"""
        if len(self.replay_buffer) == 0:
            if verbose:
                print("✗ 经验池为空，请先加载专家数据")
            return False
        
        if verbose:
            print(f"开始训练：使用已加载的 {len(self.replay_buffer)} 个经验")
            print(f"使用设备: {self.device}")
            print("="*70)
        
        # 训练网络
        if verbose:
            print("训练价值网络...")
            print(f"  网络参数量: {sum(p.numel() for p in self.value_net.parameters()):,}")
        
        self.value_net.train()
        best_epoch_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_losses = []
            
            # 多个mini-batch
            num_batches = max(len(self.replay_buffer) // batch_size, 1)
            
            for _ in range(num_batches):
                # 从经验池采样
                beta = min(1.0, 0.4 + 0.6 * (self.train_step / (epochs * num_batches)))
                samples, indices, weights = self.replay_buffer.sample(batch_size, beta)
                
                if not samples:
                    continue
                
                # 准备批量数据
                states = torch.FloatTensor([s.state for s in samples]).to(self.device)
                targets = torch.FloatTensor([[s.value] for s in samples]).to(self.device)
                weights = torch.FloatTensor(weights).to(self.device).unsqueeze(1)
                
                # 前向传播
                predictions = self.value_net(states)
                
                # 计算TD误差
                td_errors = torch.abs(predictions - targets).detach().cpu().numpy().flatten()
                
                # 更新优先级
                new_priorities = td_errors + 1e-6
                self.replay_buffer.update_priorities(indices, new_priorities)
                
                # 加权MSE损失
                loss = (weights * F.mse_loss(predictions, targets, reduction='none')).mean()
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
                self.train_step += 1
                
                # 软更新目标网络
                if self.train_step % 10 == 0:
                    self.soft_update_target()
            
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0
            self.scheduler.step(avg_loss)
            
            if verbose:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.6f}, LR={lr:.6f}")
            
            # 保存最佳模型
            if avg_loss < best_epoch_loss:
                best_epoch_loss = avg_loss
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self.save_checkpoint('best_model.pt', is_best=True)
            
            # 定期保存检查点
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
        
        if verbose:
            print(f"\n{'='*70}")
            print("✓ 训练完成！")
            print(f"  最佳损失: {self.best_loss:.6f}")
            print(f"  最终损失: {best_epoch_loss:.6f}")
            print(f"{'='*70}")
        
        return True
    
    def train_from_expert_trajectories(self, num_games=100, 
                                       timeout_per_game=30,
                                       epochs=10,
                                       batch_size=64,
                                       verbose=True) -> bool:
        if verbose:
            print(f"开始训练：从 {num_games} 个盘面生成专家轨迹")
            print(f"使用设备: {self.device}")
            print("="*70)
        
        # 阶段1：收集专家轨迹
        if verbose:
            print("阶段1：生成专家轨迹...")
        
        successful = 0
        total_states = 0
        
        for game_num in range(1, num_games + 1):
            if verbose:
                print(f"\r  生成轨迹 {game_num}/{num_games}...", end='', flush=True)
            
            status = self._create_random_layout()
            solver = BaselineSolver(status)
            solution = solver.solve(timeout=timeout_per_game)
            
            if solution:
                successful += 1
                total_states += len(solution)
                
                # 添加到经验回放
                for state, steps_to_goal in solution:
                    features = self.feature_extractor.extract(state)
                    target_value = -steps_to_goal / 50.0  # 归一化
                    
                    # 初始优先级基于距离目标的远近
                    priority = abs(target_value) + 0.1
                    self.replay_buffer.push(features, target_value, priority)
        
        if verbose:
            print(f"\n  ✓ 成功生成 {successful}/{num_games} 个轨迹")
            print(f"  ✓ 收集 {total_states} 个训练样本")
            print(f"  ✓ 经验池大小: {len(self.replay_buffer)}")
        
        if successful == 0:
            if verbose:
                print("\n✗ 训练失败：没有成功解决任何盘面")
            return False
        
        # 阶段2：训练网络
        if verbose:
            print("\n阶段2：训练价值网络...")
            print(f"  网络参数量: {sum(p.numel() for p in self.value_net.parameters()):,}")
        
        self.value_net.train()
        best_epoch_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_losses = []
            
            # 多个mini-batch
            num_batches = max(len(self.replay_buffer) // batch_size, 1)
            
            for _ in range(num_batches):
                # 从经验池采样
                beta = min(1.0, 0.4 + 0.6 * (self.train_step / (epochs * num_batches)))
                samples, indices, weights = self.replay_buffer.sample(batch_size, beta)
                
                if not samples:
                    continue
                
                # 准备批量数据
                states = torch.FloatTensor([s.state for s in samples]).to(self.device)
                targets = torch.FloatTensor([[s.value] for s in samples]).to(self.device)
                weights = torch.FloatTensor(weights).to(self.device).unsqueeze(1)
                
                # 前向传播
                predictions = self.value_net(states)
                
                # 计算TD误差
                td_errors = torch.abs(predictions - targets).detach().cpu().numpy().flatten()
                
                # 更新优先级
                new_priorities = td_errors + 1e-6
                self.replay_buffer.update_priorities(indices, new_priorities)
                
                # 加权MSE损失
                loss = (weights * F.mse_loss(predictions, targets, reduction='none')).mean()
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
                self.train_step += 1
                
                # 软更新目标网络
                if self.train_step % 10 == 0:
                    self.soft_update_target()
            
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0
            self.scheduler.step(avg_loss)
            
            if verbose:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.6f}, LR={lr:.6f}")
            
            # 保存最佳模型
            if avg_loss < best_epoch_loss:
                best_epoch_loss = avg_loss
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self.save_checkpoint('best_model.pt', is_best=True)
            
            # 定期保存检查点
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
        
        if verbose:
            print(f"\n{'='*70}")
            print("✓ 训练完成！")
            print(f"  成功率: {100*successful/num_games:.1f}%")
            print(f"  最佳损失: {self.best_loss:.6f}")
            print(f"  最终损失: {best_epoch_loss:.6f}")
            print(f"{'='*70}")
        
        return True
    
    def _create_random_layout(self) -> Status:
        s = Status()
        cards = [Card(i) for i in range(40)]
        random.shuffle(cards)
        for i in range(8):
            for j in range(5):
                s.stacks[i].push(cards[i*5 + j])
        s.auto_remove()
        return s
    
    def evaluate_value(self, status: Status) -> float:
        """评估状态价值"""
        self.value_net.eval()
        with torch.no_grad():
            features = self.feature_extractor.extract(status)
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            value = self.value_net(features_tensor).item()
        return value
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """保存检查点"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        checkpoint = {
            'value_net_state': self.value_net.state_dict(),
            'target_net_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'train_step': self.train_step,
            'best_loss': self.best_loss,
            'replay_buffer_size': len(self.replay_buffer)
        }
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            if filename != 'best_model.pt':
                print(f"  ✓ 保存最佳模型: {best_path}")
    
    def load_checkpoint(self, filename: str = 'best_model.pt') -> bool:
        """加载检查点"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        if not os.path.exists(filepath):
            print(f"✗ 检查点文件不存在: {filepath}")
            return False
        
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.value_net.load_state_dict(checkpoint['value_net_state'])
        self.target_net.load_state_dict(checkpoint['target_net_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.train_step = checkpoint['train_step']
        self.best_loss = checkpoint['best_loss']
        
        print(f"✓ 成功加载检查点: {filepath}")
        print(f"  训练步数: {self.train_step}")
        print(f"  最佳损失: {self.best_loss:.6f}")
        return True
    
    def continue_training(self, num_games=50, timeout_per_game=30, 
                         epochs=5, batch_size=64, verbose=True):
        """继续训练"""
        if verbose:
            print("\n继续训练模式")
            print("="*70)
        
        return self.train_from_expert_trajectories(
            num_games=num_games,
            timeout_per_game=timeout_per_game,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )


# ==================== 传统A*求解器 ====================

class BaselineSolver:
    """传统A*求解器"""
    
    def __init__(self, status: Status):
        self.start = status
        self.nodes_explored = 0
    
    def solve(self, timeout: float = 30) -> Optional[List[Tuple[Status, int]]]:
        start_time = time.time()
        visited = set()
        heap = []
        counter = 0
        self.nodes_explored = 0
        
        start_copy = self.start.copy()
        h = self._heuristic(start_copy)
        heapq.heappush(heap, (h, 0, counter, start_copy, []))
        visited.add(start_copy.hash_key())
        
        while heap:
            if time.time() - start_time > timeout:
                return None
            
            _, g, _, curr, path = heapq.heappop(heap)
            self.nodes_explored += 1
            
            if curr.is_solved():
                result = []
                for i, state in enumerate(path):
                    steps_to_goal = len(path) - i
                    result.append((state, steps_to_goal))
                result.append((curr, 0))
                return result
            
            for next_status in self._get_successors(curr):
                key = next_status.hash_key()
                if key in visited:
                    continue
                visited.add(key)
                
                new_h = self._heuristic(next_status)
                heapq.heappush(heap, (g + 1 + new_h, g + 1, counter := counter + 1,
                                     next_status, path + [next_status]))
        
        return None
    
    def _heuristic(self, s: Status) -> int:
        h = 0
        collected = sum(s.sorted_tops)
        h += (27 - collected) * 4
        h += 0 if s.special_removed else 20
        
        flower_cnt = [0, 0, 0]
        for st in s.stacks:
            if st.top() and st.top().type == CardType.FLOWER:
                flower_cnt[st.top().color] += 1
        for c in s.stash:
            if c.type == CardType.FLOWER:
                flower_cnt[c.color] += 1
        for cnt in flower_cnt:
            h += max(0, 4 - cnt) * 4
        
        h += len(s.stash) * 8
        empty = sum(1 for st in s.stacks if not st.cards)
        h -= empty * 12
        for st in s.stacks:
            h -= st.cont_len * st.cont_len
        
        return max(0, h)
    
    def _get_successors(self, status: Status) -> List[Status]:
        base = status.copy()
        successors = []
        
        for src_i in range(8):
            if not base.stacks[src_i].cards:
                continue
            movable = base.stacks[src_i].movable_count()
            for cnt in range(1, movable + 1):
                group_bottom = base.stacks[src_i].cards[-cnt]
                for dst_i in range(8):
                    if src_i == dst_i:
                        continue
                    dst = base.stacks[dst_i]
                    can = not dst.cards or (
                        group_bottom.type == CardType.NUMBER and dst.top() and
                        dst.top().type == CardType.NUMBER and
                        group_bottom.color != dst.top().color and
                        group_bottom.num == dst.top().num - 1)
                    if can:
                        if cnt == 1 and not dst.cards and len(base.stacks[src_i].cards) > 1:
                            if base.stacks[src_i].cont_len != len(base.stacks[src_i].cards):
                                continue
                        
                        cur = base.copy()
                        moved = [cur.stacks[src_i].pop() for _ in range(cnt)][::-1]
                        for c in moved:
                            cur.stacks[dst_i].push(c)
                        cur.auto_remove()
                        successors.append(cur)
        
        if len(base.stash) < base.stash_limit:
            for i in range(8):
                if base.stacks[i].cards:
                    top_card = base.stacks[i].top()
                    if base._can_auto_remove(top_card):
                        continue
                    
                    cur = base.copy()
                    card = cur.stacks[i].pop()
                    cur.stash.add(card)
                    cur.auto_remove()
                    successors.append(cur)
        
        for card in list(base.stash):
            for i in range(8):
                dst = base.stacks[i]
                can = not dst.cards or (
                    card.type == CardType.NUMBER and dst.top() and
                    dst.top().type == CardType.NUMBER and
                    card.color != dst.top().color and
                    card.num == dst.top().num - 1)
                if can:
                    cur = base.copy()
                    card_in_cur = next((c for c in cur.stash if c.value == card.value), None)
                    if card_in_cur:
                        cur.stash.remove(card_in_cur)
                        cur.stacks[i].push(card_in_cur)
                        cur.auto_remove()
                        successors.append(cur)
        
        return successors


# ==================== RL增强的求解器 ====================

class RLEnhancedSolver:
    """使用RL价值函数增强的A*求解器"""
    
    def __init__(self, status: Status, 
                 trainer: Optional[PyTorchRLTrainer] = None,
                 rl_weight: float = 0.5):
        self.start = status
        self.trainer = trainer
        self.rl_weight = rl_weight
        self.nodes_explored = 0
    
    def solve(self, timeout: float = 180, verbose: bool = False) -> Optional[List[str]]:
        start_time = time.time()
        visited = set()
        heap = []
        counter = 0
        self.nodes_explored = 0
        
        start_copy = self.start.copy()
        h = self._heuristic(start_copy)
        heapq.heappush(heap, (h, 0, counter, start_copy, []))
        visited.add(start_copy.hash_key())
        
        last_print_time = start_time
        
        while heap:
            current_time = time.time()
            
            if current_time - start_time > timeout:
                if verbose:
                    print(f"\n✗ 超时 ({timeout:.0f}秒)")
                return None
            
            if verbose and current_time - last_print_time > 2:
                elapsed = current_time - start_time
                print(f"\r  探索节点: {self.nodes_explored}, 队列: {len(heap)}, 用时: {elapsed:.1f}s", 
                      end='', flush=True)
                last_print_time = current_time
            
            _, g, _, curr, path = heapq.heappop(heap)
            self.nodes_explored += 1
            
            if curr.is_solved():
                if verbose:
                    print()
                return path
            
            for desc, next_status in self._get_successors(curr):
                key = next_status.hash_key()
                if key in visited:
                    continue
                visited.add(key)
                
                new_h = self._heuristic(next_status)
                heapq.heappush(heap, (g + 1 + new_h, g + 1, counter := counter + 1,
                                     next_status, path + [desc]))
        
        if verbose:
            print(f"\n✗ 未找到解")
        return None
    
    def _heuristic(self, s: Status) -> float:
        # 传统启发式
        h_traditional = 0
        collected = sum(s.sorted_tops)
        h_traditional += (27 - collected) * 3
        h_traditional += 0 if s.special_removed else 15
        
        flower_cnt = [0, 0, 0]
        for st in s.stacks:
            if st.top() and st.top().type == CardType.FLOWER:
                flower_cnt[st.top().color] += 1
        for c in s.stash:
            if c.type == CardType.FLOWER:
                flower_cnt[c.color] += 1
        for cnt in flower_cnt:
            h_traditional += max(0, 4 - cnt) * 3
        
        h_traditional += len(s.stash) * 6
        empty = sum(1 for st in s.stacks if not st.cards)
        h_traditional -= empty * 10
        for st in s.stacks:
            h_traditional -= st.cont_len * st.cont_len
        
        # 混合RL价值
        if self.trainer:
            rl_value = self.trainer.evaluate_value(s)
            h_rl = -rl_value * 50  # 转换为正的启发值
            h = (1 - self.rl_weight) * h_traditional + self.rl_weight * h_rl
        else:
            h = h_traditional
        
        return max(0, h)
    
    def _get_successors(self, status: Status) -> List[Tuple[str, Status]]:
        base = status.copy()
        successors = []
        
        for src_i in range(8):
            if not base.stacks[src_i].cards:
                continue
            movable = base.stacks[src_i].movable_count()
            for cnt in range(1, movable + 1):
                group_bottom = base.stacks[src_i].cards[-cnt]
                for dst_i in range(8):
                    if src_i == dst_i:
                        continue
                    dst = base.stacks[dst_i]
                    can = not dst.cards or (
                        group_bottom.type == CardType.NUMBER and dst.top() and
                        dst.top().type == CardType.NUMBER and
                        group_bottom.color != dst.top().color and
                        group_bottom.num == dst.top().num - 1)
                    if can:
                        if cnt == 1 and not dst.cards and len(base.stacks[src_i].cards) > 1:
                            if base.stacks[src_i].cont_len != len(base.stacks[src_i].cards):
                                continue
                        
                        cur = base.copy()
                        moved = [cur.stacks[src_i].pop() for _ in range(cnt)][::-1]
                        for c in moved:
                            cur.stacks[dst_i].push(c)
                        cur.auto_remove()
                        desc = f"列{src_i+1} → 列{dst_i+1} ({cnt}张)"
                        successors.append((desc, cur))
        
        if len(base.stash) < base.stash_limit:
            for i in range(8):
                if base.stacks[i].cards:
                    top_card = base.stacks[i].top()
                    if base._can_auto_remove(top_card):
                        continue
                    
                    cur = base.copy()
                    card = cur.stacks[i].pop()
                    cur.stash.add(card)
                    cur.auto_remove()
                    desc = f"列{i+1} → Stash ({card})"
                    successors.append((desc, cur))
        
        for card in list(base.stash):
            for i in range(8):
                dst = base.stacks[i]
                can = not dst.cards or (
                    card.type == CardType.NUMBER and dst.top() and
                    dst.top().type == CardType.NUMBER and
                    card.color != dst.top().color and
                    card.num == dst.top().num - 1)
                if can:
                    cur = base.copy()
                    card_in_cur = next((c for c in cur.stash if c.value == card.value), None)
                    if card_in_cur:
                        cur.stash.remove(card_in_cur)
                        cur.stacks[i].push(card_in_cur)
                        cur.auto_remove()
                        desc = f"Stash ({card}) → 列{i+1}"
                        successors.append((desc, cur))
        
        return successors


# ==================== 工具函数 ====================

def create_random_layout(seed: Optional[int] = None) -> Status:
    if seed is not None:
        random.seed(seed)
    s = Status()
    cards = [Card(i) for i in range(40)]
    random.shuffle(cards)
    for i in range(8):
        for j in range(5):
            s.stacks[i].push(cards[i*5 + j])
    s.auto_remove()
    return s


# ==================== 主程序 ====================

def main():
    print("="*70)
    print(" " * 12 + "Shenzhen Solitaire 求解器 v7.0 (PyTorch)")
    print("="*70)
    print("\n特性:")
    print("  ✓ PyTorch深度学习框架")
    print("  ✓ 残差网络 + 注意力机制")
    print("  ✓ 优先级经验回放")
    print("  ✓ 目标网络 + 软更新")
    print("  ✓ 完善的checkpoint机制")
    print("  ✓ 增强的128维特征")
    print("  ✓ 支持加载预采集专家数据")
    print("\n" + "="*70)
    print("\n选择模式:")
    print("  1. 训练新模型（实时生成数据）")
    print("  2. 从专家数据文件训练")
    print("  3. 继续训练现有模型")
    print("  4. 使用训练好的模型求解")
    print("  5. 对比测试：传统 vs RL增强")
    print("  6. 快速演示")
    print("="*70)
    
    choice = input("\n请选择 (1-6): ").strip()
    
    if choice == "1":
        # 训练新模型
        print("\n" + "="*70)
        print("模式1：训练新模型")
        print("="*70)
        
        num_games = int(input("训练盘面数量 (推荐100-200): ") or "100")
        timeout = float(input("每个盘面A*求解时限/秒 (推荐30): ") or "30")
        epochs = int(input("训练轮数 (推荐10-20): ") or "10")
        lr = float(input("学习率 (推荐0.0003): ") or "0.0003")
        hidden = int(input("隐藏层大小 (推荐256): ") or "256")
        
        print(f"\n开始训练...")
        
        trainer = PyTorchRLTrainer(learning_rate=lr, hidden_size=hidden)
        success = trainer.train_from_expert_trajectories(
            num_games=num_games,
            timeout_per_game=timeout,
            epochs=epochs,
            batch_size=64,
            verbose=True
        )
        
        if success:
            test = input("\n是否测试训练后的模型？(y/n): ").strip().lower()
            if test == 'y':
                print("\n" + "="*70)
                print("测试训练后的模型")
                print("="*70)
                
                test_status = create_random_layout()
                print("\n测试盘面:")
                test_status.print_status()
                
                print("使用RL增强求解器 (权重=0.5)...")
                solver = RLEnhancedSolver(test_status, trainer, rl_weight=0.5)
                solution = solver.solve(timeout=60, verbose=True)
                
                if solution:
                    print(f"\n✓ 找到解法！步数: {len(solution)}, 探索: {solver.nodes_explored}")
                else:
                    print(f"\n✗ 未在60秒内找到解法")
    
    elif choice == "2":
        # 从专家数据文件训练
        print("\n" + "="*70)
        print("模式2：从专家数据文件训练")
        print("="*70)
        
        trainer = PyTorchRLTrainer()
        
        # 列出可用的专家数据文件
        files = trainer.list_expert_data_files()
        
        if not files:
            print("\n未找到任何专家数据文件")
            print("请先运行 collect_expert_data.py 采集专家数据")
            return
        
        print("\n可用的专家数据文件:\n")
        for i, f in enumerate(files, 1):
            filepath = os.path.join(trainer.expert_data_dir, f)
            size = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  {i}. {f} ({size:.2f} MB)")
        
        file_num = int(input("\n选择文件编号: ").strip())
        
        if 1 <= file_num <= len(files):
            filepath = os.path.join(trainer.expert_data_dir, files[file_num - 1])
            
            # 加载专家数据
            if trainer.load_expert_data(filepath):
                # 训练参数
                epochs = int(input("\n训练轮数 (推荐10-20): ") or "10")
                lr = float(input("学习率 (推荐0.0003): ") or "0.0003")
                
                # 更新学习率
                for param_group in trainer.optimizer.param_groups:
                    param_group['lr'] = lr
                
                # 开始训练
                trainer.train_from_loaded_data(
                    epochs=epochs,
                    batch_size=64,
                    verbose=True
                )
                
                # 测试
                test = input("\n是否测试训练后的模型？(y/n): ").strip().lower()
                if test == 'y':
                    print("\n" + "="*70)
                    print("测试训练后的模型")
                    print("="*70)
                    
                    test_status = create_random_layout()
                    print("\n测试盘面:")
                    test_status.print_status()
                    
                    print("使用RL增强求解器 (权重=0.5)...")
                    solver = RLEnhancedSolver(test_status, trainer, rl_weight=0.5)
                    solution = solver.solve(timeout=60, verbose=True)
                    
                    if solution:
                        print(f"\n✓ 找到解法！步数: {len(solution)}, 探索: {solver.nodes_explored}")
                    else:
                        print(f"\n✗ 未在60秒内找到解法")
        else:
            print("无效的文件编号")
    
    elif choice == "3":
        # 继续训练
        print("\n" + "="*70)
        print("模式3：继续训练")
        print("="*70)
        
        trainer = PyTorchRLTrainer()
        if not trainer.load_checkpoint():
            print("未找到检查点，将创建新模型")
        
        num_games = int(input("\n额外训练盘面数量 (推荐50-100): ") or "50")
        timeout = float(input("每个盘面A*求解时限/秒 (推荐30): ") or "30")
        epochs = int(input("训练轮数 (推荐5-10): ") or "5")
        
        trainer.continue_training(
            num_games=num_games,
            timeout_per_game=timeout,
            epochs=epochs,
            verbose=True
        )
    
    elif choice == "4":
        # 使用模型求解
        print("\n" + "="*70)
        print("模式4：使用模型求解")
        print("="*70)
        
        trainer = PyTorchRLTrainer()
        if not trainer.load_checkpoint():
            print("未找到模型，请先训练")
            return
        
        seed_input = input("\n输入随机种子（留空则完全随机）: ").strip()
        status = create_random_layout(int(seed_input) if seed_input else None)
        
        print("\n初始盘面:")
        status.print_status()
        
        value = trainer.evaluate_value(status)
        print(f"RL价值评估: {value:.4f}\n")
        
        rl_weight = float(input("RL启发式权重 0-1 (推荐0.5): ") or "0.5")
        
        print(f"\n开始求解 (RL权重={rl_weight})...")
        start_time = time.time()
        solver = RLEnhancedSolver(status, trainer, rl_weight=rl_weight)
        solution = solver.solve(timeout=180, verbose=True)
        elapsed = time.time() - start_time
        
        if solution:
            print(f"\n✓ 找到解法！")
            print(f"  步数: {len(solution)}")
            print(f"  探索节点: {solver.nodes_explored}")
            print(f"  用时: {elapsed:.2f} 秒")
    
    elif choice == "5":
        # 对比测试
        print("\n" + "="*70)
        print("模式5：对比测试")
        print("="*70)
        
        trainer = PyTorchRLTrainer()
        if not trainer.load_checkpoint():
            print("未找到模型，将仅测试传统方法")
            trainer = None
        
        num_tests = int(input("\n测试盘面数量 (推荐10-20): ") or "10")
        timeout = float(input("每个盘面求解时限/秒 (推荐60): ") or "60")
        
        results = {
            'traditional': {'solved': 0, 'steps': [], 'nodes': [], 'times': []},
            'rl': {'solved': 0, 'steps': [], 'nodes': [], 'times': []}
        }
        
        print(f"\n{'='*70}")
        print(f"开始测试 {num_tests} 个盘面")
        print(f"{'='*70}\n")
        
        for test_num in range(1, num_tests + 1):
            print(f"测试 {test_num}/{num_tests}")
            print("-"*70)
            
            status = create_random_layout(seed=2000 + test_num)
            
            # 传统方法
            print("  传统A*...", end=' ', flush=True)
            start = time.time()
            solver_trad = RLEnhancedSolver(status.copy(), trainer=None)
            sol_trad = solver_trad.solve(timeout=timeout, verbose=False)
            time_trad = time.time() - start
            
            if sol_trad:
                results['traditional']['solved'] += 1
                results['traditional']['steps'].append(len(sol_trad))
                results['traditional']['nodes'].append(solver_trad.nodes_explored)
                results['traditional']['times'].append(time_trad)
                print(f"✓ ({len(sol_trad)}步, {solver_trad.nodes_explored}节点, {time_trad:.1f}s)")
            else:
                print(f"✗ 超时")
            
            # RL方法
            if trainer:
                print("  RL增强...", end=' ', flush=True)
                start = time.time()
                solver_rl = RLEnhancedSolver(status.copy(), trainer, rl_weight=0.5)
                sol_rl = solver_rl.solve(timeout=timeout, verbose=False)
                time_rl = time.time() - start
                
                if sol_rl:
                    results['rl']['solved'] += 1
                    results['rl']['steps'].append(len(sol_rl))
                    results['rl']['nodes'].append(solver_rl.nodes_explored)
                    results['rl']['times'].append(time_rl)
                    print(f"✓ ({len(sol_rl)}步, {solver_rl.nodes_explored}节点, {time_rl:.1f}s)")
                else:
                    print(f"✗ 超时")
            
            print()
        
        # 显示结果
        print("="*70)
        print("对比结果")
        print("="*70)
        
        def print_stats(name, data):
            print(f"\n{name}:")
            solved = data['solved']
            print(f"  解决率: {solved}/{num_tests} ({100*solved/num_tests:.1f}%)")
            if solved > 0:
                print(f"  平均步数: {sum(data['steps'])/solved:.1f}")
                print(f"  平均探索节点: {sum(data['nodes'])/solved:.0f}")
                print(f"  平均用时: {sum(data['times'])/solved:.2f}秒")
        
        print_stats("传统A*", results['traditional'])
        if trainer:
            print_stats("RL增强", results['rl'])
            
            if results['traditional']['solved'] > 0 and results['rl']['solved'] > 0:
                avg_nodes_trad = sum(results['traditional']['nodes']) / results['traditional']['solved']
                avg_nodes_rl = sum(results['rl']['nodes']) / results['rl']['solved']
                improvement = (avg_nodes_trad - avg_nodes_rl) / avg_nodes_trad * 100
                print(f"\n改进分析:")
                print(f"  节点探索减少: {improvement:+.1f}%")
    
    elif choice == "6":
        # 快速演示
        print("\n" + "="*70)
        print("模式6：快速演示")
        print("="*70)
        
        print("\n快速训练（10个盘面，5轮）...")
        trainer = PyTorchRLTrainer()
        trainer.train_from_expert_trajectories(
            num_games=10, timeout_per_game=20, epochs=5, verbose=True
        )
        
        print("\n测试模型...")
        for i in range(3):
            print(f"\n测试 {i+1}/3")
            status = create_random_layout(seed=3000 + i)
            solver = RLEnhancedSolver(status, trainer, rl_weight=0.5)
            solution = solver.solve(timeout=30, verbose=False)
            if solution:
                print(f"  ✓ 步数:{len(solution)}, 节点:{solver.nodes_explored}")
            else:
                print(f"  ✗ 超时")
    
    else:
        print("\n✗ 无效的选择")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已中断")
    except Exception as e:
        print(f"\n\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
