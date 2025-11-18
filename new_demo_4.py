# -*- coding: utf-8 -*-
"""
Shenzhen Solitaire 求解器 v6.0
完整的强化学习实现，包含：
1. 传统A*基线求解器
2. 从专家轨迹学习的价值网络
3. RL增强的启发式搜索
4. 完整的训练和评估流程
"""

import heapq
import time
import hashlib
import random
import numpy as np
import pickle
from typing import List, Optional, Tuple, Dict, Set
from enum import Enum
from collections import deque
from dataclasses import dataclass

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
            # 紧凑显示
            print(f"Stash:{len(self.stash)}/{self.stash_limit} | 已收:R{self.sorted_tops[0]}G{self.sorted_tops[1]}B{self.sorted_tops[2]}SP{int(self.special_removed)}")
        else:
            # 详细显示
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
        """生成规范化的状态哈希"""
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
        """检查卡牌是否可以自动收集"""
        if card.type == CardType.SPECIAL:
            return True
        if card.type != CardType.NUMBER:
            return False
        
        color, need = card.color, card.num
        top = self.sorted_tops[color]
        
        if top + 1 == need:
            if need <= 2:
                return True
            # 需要其他颜色至少到达 top-1
            other_min = min(self.sorted_tops[(color+1)%3], self.sorted_tops[(color+2)%3])
            return other_min >= top - 1
        return False

    def _auto_remove_flowers(self) -> bool:
        """自动移除凑齐的四张花牌"""
        count = [0, 0, 0]
        sources = []
        
        # 统计花牌
        for i, st in enumerate(self.stacks):
            t = st.top()
            if t and t.type == CardType.FLOWER:
                count[t.color] += 1
                sources.append((True, i, t))
        
        for c in self.stash:
            if c.type == CardType.FLOWER:
                count[c.color] += 1
                sources.append((False, -1, c))
        
        # 检查是否可以移除
        for color in range(3):
            if count[color] == 4:
                # 需要有stash空位或stash中有同色花牌
                has_slot = len(self.stash) < self.stash_limit
                has_same_in_stash = any(c.type == CardType.FLOWER and c.color == color for c in self.stash)
                
                if has_slot or has_same_in_stash:
                    # 移除该颜色的所有花牌
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
        """自动移除可收集的牌"""
        changed = True
        while changed:
            changed = False
            
            # 检查栈顶
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
            
            # 检查stash
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
            
            # 检查花牌
            if self._auto_remove_flowers():
                changed = True

    def is_solved(self) -> bool:
        """检查是否已解决"""
        if not self.special_removed:
            return False
        for st in self.stacks:
            if st.cards:
                return False
        # stash中只能有花牌
        for c in self.stash:
            if c.type != CardType.FLOWER:
                return False
        return True

    def copy(self) -> 'Status':
        """深拷贝状态"""
        new_s = Status()
        for i, st in enumerate(self.stacks):
            new_s.stacks[i].cards = st.cards.copy()
            new_s.stacks[i].cont_len = st.cont_len
        new_s.stash = {Card(c.value) for c in self.stash}
        new_s.stash_limit = self.stash_limit
        new_s.sorted_tops = self.sorted_tops[:]
        new_s.special_removed = self.special_removed
        return new_s


# ==================== 特征提取器 ====================

class FeatureExtractor:
    """从游戏状态提取神经网络特征"""
    
    @staticmethod
    def extract(status: Status) -> np.ndarray:
        """
        提取64维特征向量：
        - 8列 × 6特征 = 48
        - Stash: 6特征
        - 收集进度: 4特征
        - 全局: 6特征
        """
        features = []
        
        # 每列信息 (48维)
        for st in status.stacks:
            # 列高度
            features.append(len(st.cards) / 5.0)
            # 连续牌长度
            features.append(st.cont_len / 5.0)
            # 顶牌信息
            top = st.top()
            if top and top.type == CardType.NUMBER:
                features.append(top.color / 2.0)
                features.append(top.num / 9.0)
            else:
                features.extend([0.0, 0.0])
            # 是否为空列
            features.append(1.0 if not st.cards else 0.0)
            # 数字牌数量
            features.append(sum(1 for c in st.cards if c.type == CardType.NUMBER) / 5.0)
        
        # Stash信息 (6维)
        features.append(len(status.stash) / 3.0)
        features.append(status.stash_limit / 3.0)
        features.append(sum(1 for c in status.stash if c.type == CardType.NUMBER) / 3.0)
        features.append(sum(1 for c in status.stash if c.type == CardType.FLOWER) / 3.0)
        features.append(1.0 if len(status.stash) >= status.stash_limit else 0.0)
        flower_in_stash = [sum(1 for c in status.stash if c.type == CardType.FLOWER and c.color == i) for i in range(3)]
        features.append(max(flower_in_stash) / 4.0 if flower_in_stash else 0.0)
        
        # 收集进度 (4维)
        features.extend([x / 9.0 for x in status.sorted_tops])
        features.append(1.0 if status.special_removed else 0.0)
        
        # 全局统计 (6维)
        total_cards = sum(len(st.cards) for st in status.stacks) + len(status.stash)
        features.append(total_cards / 40.0)
        features.append(sum(status.sorted_tops) / 27.0)
        empty_cols = sum(1 for st in status.stacks if not st.cards)
        features.append(empty_cols / 8.0)
        total_cont = sum(st.cont_len for st in status.stacks)
        features.append(total_cont / 40.0)
        progress = (sum(status.sorted_tops) + int(status.special_removed) * 3) / 30.0
        features.append(progress)
        features.append(sum(flower_in_stash) / 12.0)
        
        return np.array(features, dtype=np.float32)


# ==================== 价值网络 ====================

class ValueNetwork:
    """简单的3层全连接神经网络"""
    
    def __init__(self, input_size: int, hidden_size: int = 128):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Xavier初始化
        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(hidden_size)
        self.w3 = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
        self.b3 = np.zeros(1)
        
        # 用于Adam优化器的动量
        self.m_w1 = np.zeros_like(self.w1)
        self.v_w1 = np.zeros_like(self.w1)
        self.m_b1 = np.zeros_like(self.b1)
        self.v_b1 = np.zeros_like(self.b1)
        
        self.m_w2 = np.zeros_like(self.w2)
        self.v_w2 = np.zeros_like(self.w2)
        self.m_b2 = np.zeros_like(self.b2)
        self.v_b2 = np.zeros_like(self.b2)
        
        self.m_w3 = np.zeros_like(self.w3)
        self.v_w3 = np.zeros_like(self.w3)
        self.m_b3 = np.zeros_like(self.b3)
        self.v_b3 = np.zeros_like(self.b3)
        
        self.t = 0  # timestep for Adam
        
    def predict(self, state: np.ndarray) -> float:
        """前向传播"""
        # Layer 1
        z1 = np.dot(state, self.w1) + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        
        # Layer 2
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = np.maximum(0, z2)  # ReLU
        
        # Output
        value = np.dot(a2, self.w3) + self.b3
        return float(value[0])
    
    def train_step(self, state: np.ndarray, target: float, lr: float = 0.001):
        """使用Adam优化器的训练步骤"""
        self.t += 1
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        
        # 前向传播
        z1 = np.dot(state, self.w1) + self.b1
        a1 = np.maximum(0, z1)
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = np.maximum(0, z2)
        output = np.dot(a2, self.w3) + self.b3
        
        # 反向传播
        delta3 = (output - target).reshape(-1)
        dw3 = np.outer(a2, delta3)
        db3 = delta3
        
        delta2 = np.dot(delta3, self.w3.T) * (z2 > 0)
        dw2 = np.outer(a1, delta2)
        db2 = delta2
        
        delta1 = np.dot(delta2, self.w2.T) * (z1 > 0)
        dw1 = np.outer(state, delta1)
        db1 = delta1
        
        # Adam更新 - Layer 3
        self.m_w3 = beta1 * self.m_w3 + (1 - beta1) * dw3
        self.v_w3 = beta2 * self.v_w3 + (1 - beta2) * (dw3 ** 2)
        m_hat_w3 = self.m_w3 / (1 - beta1 ** self.t)
        v_hat_w3 = self.v_w3 / (1 - beta2 ** self.t)
        self.w3 -= lr * m_hat_w3 / (np.sqrt(v_hat_w3) + eps)
        
        self.m_b3 = beta1 * self.m_b3 + (1 - beta1) * db3
        self.v_b3 = beta2 * self.v_b3 + (1 - beta2) * (db3 ** 2)
        m_hat_b3 = self.m_b3 / (1 - beta1 ** self.t)
        v_hat_b3 = self.v_b3 / (1 - beta2 ** self.t)
        self.b3 -= lr * m_hat_b3 / (np.sqrt(v_hat_b3) + eps)
        
        # Adam更新 - Layer 2
        self.m_w2 = beta1 * self.m_w2 + (1 - beta1) * dw2
        self.v_w2 = beta2 * self.v_w2 + (1 - beta2) * (dw2 ** 2)
        m_hat_w2 = self.m_w2 / (1 - beta1 ** self.t)
        v_hat_w2 = self.v_w2 / (1 - beta2 ** self.t)
        self.w2 -= lr * m_hat_w2 / (np.sqrt(v_hat_w2) + eps)
        
        self.m_b2 = beta1 * self.m_b2 + (1 - beta1) * db2
        self.v_b2 = beta2 * self.v_b2 + (1 - beta2) * (db2 ** 2)
        m_hat_b2 = self.m_b2 / (1 - beta1 ** self.t)
        v_hat_b2 = self.v_b2 / (1 - beta2 ** self.t)
        self.b2 -= lr * m_hat_b2 / (np.sqrt(v_hat_b2) + eps)
        
        # Adam更新 - Layer 1
        self.m_w1 = beta1 * self.m_w1 + (1 - beta1) * dw1
        self.v_w1 = beta2 * self.v_w1 + (1 - beta2) * (dw1 ** 2)
        m_hat_w1 = self.m_w1 / (1 - beta1 ** self.t)
        v_hat_w1 = self.v_w1 / (1 - beta2 ** self.t)
        self.w1 -= lr * m_hat_w1 / (np.sqrt(v_hat_w1) + eps)
        
        self.m_b1 = beta1 * self.m_b1 + (1 - beta1) * db1
        self.v_b1 = beta2 * self.v_b1 + (1 - beta2) * (db1 ** 2)
        m_hat_b1 = self.m_b1 / (1 - beta1 ** self.t)
        v_hat_b1 = self.v_b1 / (1 - beta2 ** self.t)
        self.b1 -= lr * m_hat_b1 / (np.sqrt(v_hat_b1) + eps)
    
    def save(self, filename: str):
        """保存模型"""
        data = {
            'w1': self.w1, 'b1': self.b1,
            'w2': self.w2, 'b2': self.b2,
            'w3': self.w3, 'b3': self.b3,
            'm_w1': self.m_w1, 'v_w1': self.v_w1, 'm_b1': self.m_b1, 'v_b1': self.v_b1,
            'm_w2': self.m_w2, 'v_w2': self.v_w2, 'm_b2': self.m_b2, 'v_b2': self.v_b2,
            'm_w3': self.m_w3, 'v_w3': self.v_w3, 'm_b3': self.m_b3, 'v_b3': self.v_b3,
            't': self.t
        }
        np.savez(filename, **data)
        print(f"✓ 模型已保存到 {filename}")
    
    def load(self, filename: str):
        """加载模型"""
        data = np.load(filename)
        self.w1 = data['w1']
        self.b1 = data['b1']
        self.w2 = data['w2']
        self.b2 = data['b2']
        self.w3 = data['w3']
        self.b3 = data['b3']
        
        if 'm_w1' in data:
            self.m_w1 = data['m_w1']
            self.v_w1 = data['v_w1']
            self.m_b1 = data['m_b1']
            self.v_b1 = data['v_b1']
            self.m_w2 = data['m_w2']
            self.v_w2 = data['v_w2']
            self.m_b2 = data['m_b2']
            self.v_b2 = data['v_b2']
            self.m_w3 = data['m_w3']
            self.v_w3 = data['v_w3']
            self.m_b3 = data['m_b3']
            self.v_b3 = data['v_b3']
            self.t = int(data['t'])
        
        print(f"✓ 模型已从 {filename} 加载")


# ==================== 传统A*求解器 ====================

class BaselineSolver:
    """传统A*求解器，用于生成训练数据和作为基线"""
    
    def __init__(self, status: Status):
        self.start = status
        self.nodes_explored = 0
    
    def solve(self, timeout: float = 30) -> Optional[List[Tuple[Status, int]]]:
        """
        求解并返回完整路径
        返回: List[(状态, 到目标的步数)] 或 None
        """
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
                # 构建结果：每个状态对应到目标的剩余步数
                result = []
                for i, state in enumerate(path):
                    steps_to_goal = len(path) - i
                    result.append((state, steps_to_goal))
                result.append((curr, 0))  # 终点
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
        """传统启发函数"""
        h = 0
        
        # 未收集的数字牌
        collected = sum(s.sorted_tops)
        h += (27 - collected) * 4
        
        # 特殊牌惩罚
        if not s.special_removed:
            h += 20
        
        # 花牌不足
        flower_cnt = [0, 0, 0]
        for st in s.stacks:
            if st.top() and st.top().type == CardType.FLOWER:
                flower_cnt[st.top().color] += 1
        for c in s.stash:
            if c.type == CardType.FLOWER:
                flower_cnt[c.color] += 1
        for cnt in flower_cnt:
            h += max(0, 4 - cnt) * 4
        
        # Stash惩罚
        h += len(s.stash) * 8
        
        # 空列奖励
        empty = sum(1 for st in s.stacks if not st.cards)
        h -= empty * 12
        
        # 连续牌奖励
        for st in s.stacks:
            h -= st.cont_len * st.cont_len
        
        # 深度惩罚
        for color in range(3):
            need = s.sorted_tops[color] + 1
            if need > 9:
                continue
            depth = 99
            for st in s.stacks:
                for d, c in enumerate(reversed(st.cards), 1):
                    if c.type == CardType.NUMBER and c.color == color and c.num == need:
                        depth = min(depth, d)
                        break
            for c in s.stash:
                if c.type == CardType.NUMBER and c.color == color and c.num == need:
                    depth = min(depth, 1)
            
            if depth < 99:
                h += depth * 10
            else:
                h += 30
        
        return max(0, h)
    
    def _get_successors(self, status: Status) -> List[Status]:
        """生成所有后继状态"""
        base = status.copy()
        successors = []
        
        # Stack -> Stack
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
                        # 剪枝：单卡到空列通常不好
                        if cnt == 1 and not dst.cards and len(base.stacks[src_i].cards) > 1:
                            if base.stacks[src_i].cont_len != len(base.stacks[src_i].cards):
                                continue
                        
                        cur = base.copy()
                        moved = [cur.stacks[src_i].pop() for _ in range(cnt)][::-1]
                        for c in moved:
                            cur.stacks[dst_i].push(c)
                        cur.auto_remove()
                        successors.append(cur)
        
        # Stack -> Stash
        if len(base.stash) < base.stash_limit:
            for i in range(8):
                if base.stacks[i].cards:
                    top_card = base.stacks[i].top()
                    # 剪枝：不要把能auto_remove的牌放进stash
                    if base._can_auto_remove(top_card):
                        continue
                    
                    cur = base.copy()
                    card = cur.stacks[i].pop()
                    cur.stash.add(card)
                    cur.auto_remove()
                    successors.append(cur)
        
        # Stash -> Stack
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


# ==================== RL训练器 ====================

class RLTrainer:
    """从专家轨迹训练价值函数"""
    
    def __init__(self, learning_rate: float = 0.001):
        self.feature_extractor = FeatureExtractor()
        state_size = len(self.feature_extractor.extract(Status()))
        self.value_network = ValueNetwork(state_size, hidden_size=128)
        self.lr = learning_rate
    
    def train_from_expert_trajectories(self, num_games: int = 50, 
                                       timeout_per_game: float = 30,
                                       verbose: bool = True) -> bool:
        """从A*生成的专家轨迹中学习"""
        if verbose:
            print(f"开始训练：从 {num_games} 个盘面生成专家轨迹")
            print("="*60)
        
        successful = 0
        total_states = 0
        training_data = []
        
        # 第一阶段：收集数据
        if verbose:
            print("阶段1：生成专家轨迹...")
        
        for game_num in range(1, num_games + 1):
            if verbose:
                print(f"\r  生成轨迹 {game_num}/{num_games}...", end='', flush=True)
            
            # 创建随机盘面
            status = self._create_random_layout()
            
            # 用A*求解
            solver = BaselineSolver(status)
            solution = solver.solve(timeout=timeout_per_game)
            
            if solution:
                successful += 1
                total_states += len(solution)
                
                # 收集状态-价值对
                for state, steps_to_goal in solution:
                    features = self.feature_extractor.extract(state)
                    # 目标值：负的剩余步数（归一化）
                    target_value = -steps_to_goal / 100.0
                    training_data.append((features, target_value))
        
        if verbose:
            print(f"\n  ✓ 成功生成 {successful}/{num_games} 个轨迹")
            print(f"  ✓ 收集 {total_states} 个训练样本")
        
        if successful == 0:
            if verbose:
                print("\n✗ 训练失败：没有成功解决任何盘面")
            return False
        
        # 第二阶段：训练网络
        if verbose:
            print("\n阶段2：训练价值网络...")
        
        # 多轮训练（epoch）
        epochs = 5
        batch_size = 32
        
        for epoch in range(epochs):
            # 打乱数据
            random.shuffle(training_data)
            total_loss = 0
            
            # 批量训练
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i+batch_size]
                for features, target in batch:
                    self.value_network.train_step(features, target, self.lr)
                    # 计算损失（用于显示）
                    pred = self.value_network.predict(features)
                    total_loss += (pred - target) ** 2
            
            avg_loss = total_loss / len(training_data)
            if verbose:
                print(f"  Epoch {epoch+1}/{epochs}: 平均损失 = {avg_loss:.6f}")
        
        if verbose:
            print(f"\n{'='*60}")
            print("✓ 训练完成！")
            print(f"  成功率: {100*successful/num_games:.1f}%")
            print(f"  训练样本: {total_states} 个状态")
            print(f"  平均轨迹长度: {total_states/successful:.1f}" if successful > 0 else "")
            print(f"{'='*60}")
        
        return True
    
    def _create_random_layout(self) -> Status:
        """创建随机布局"""
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
        features = self.feature_extractor.extract(status)
        return self.value_network.predict(features)


# ==================== RL增强的求解器 ====================

class RLEnhancedSolver:
    """使用RL价值函数增强的A*求解器"""
    
    def __init__(self, status: Status, 
                 trainer: Optional[RLTrainer] = None,
                 rl_weight: float = 0.3):
        self.start = status
        self.trainer = trainer
        self.rl_weight = rl_weight
        self.nodes_explored = 0
    
    def solve(self, timeout: float = 180, verbose: bool = False) -> Optional[List[str]]:
        """
        求解并返回动作序列
        返回: List[str] 描述每步的动作
        """
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
            
            # 超时检查
            if current_time - start_time > timeout:
                if verbose:
                    print(f"\n✗ 超时 ({timeout:.0f}秒)")
                return None
            
            # 进度显示
            if verbose and current_time - last_print_time > 2:
                elapsed = current_time - start_time
                print(f"\r  探索节点: {self.nodes_explored}, 队列: {len(heap)}, 用时: {elapsed:.1f}s", 
                      end='', flush=True)
                last_print_time = current_time
            
            _, g, _, curr, path = heapq.heappop(heap)
            self.nodes_explored += 1
            
            # 检查是否解决
            if curr.is_solved():
                if verbose:
                    print()  # 换行
                return path
            
            # 生成后继
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
        """混合启发函数：传统 + RL"""
        # 传统启发式
        h_traditional = 0
        collected = sum(s.sorted_tops)
        h_traditional += (27 - collected) * 4
        h_traditional += 0 if s.special_removed else 20
        
        flower_cnt = [0, 0, 0]
        for st in s.stacks:
            if st.top() and st.top().type == CardType.FLOWER:
                flower_cnt[st.top().color] += 1
        for c in s.stash:
            if c.type == CardType.FLOWER:
                flower_cnt[c.color] += 1
        for cnt in flower_cnt:
            h_traditional += max(0, 4 - cnt) * 4
        
        h_traditional += len(s.stash) * 8
        empty = sum(1 for st in s.stacks if not st.cards)
        h_traditional -= empty * 12
        for st in s.stacks:
            h_traditional -= st.cont_len * st.cont_len
        
        # 混合RL价值
        if self.trainer:
            rl_value = self.trainer.evaluate_value(s)
            # RL输出负值（-steps/100），转换为正的启发值
            h_rl = -rl_value * 100
            h = (1 - self.rl_weight) * h_traditional + self.rl_weight * h_rl
        else:
            h = h_traditional
        
        return max(0, h)
    
    def _get_successors(self, status: Status) -> List[Tuple[str, Status]]:
        """生成后继状态和对应的动作描述"""
        base = status.copy()
        successors = []
        
        # Stack -> Stack
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
        
        # Stack -> Stash
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
        
        # Stash -> Stack
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
    """创建随机布局"""
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
    print(" " * 15 + "Shenzhen Solitaire 求解器 v6.0")
    print("="*70)
    print("\n特性:")
    print("  ✓ 传统A*基线求解器")
    print("  ✓ 从专家轨迹学习的价值网络")
    print("  ✓ RL增强的启发式搜索")
    print("  ✓ 完整的训练和评估流程")
    print("\n" + "="*70)
    print("\n选择模式:")
    print("  1. 训练RL价值网络（从专家轨迹学习）")
    print("  2. 使用训练好的RL模型求解")
    print("  3. 使用传统A*求解（不使用RL）")
    print("  4. 对比测试：传统 vs RL增强")
    print("  5. 快速演示（少量训练+测试）")
    print("="*70)
    
    choice = input("\n请选择 (1-5): ").strip()
    
    if choice == "1":
        # 训练模式
        print("\n" + "="*70)
        print("模式1：训练RL价值网络")
        print("="*70)
        print("\n说明：")
        print("  - 程序会生成随机盘面")
        print("  - 使用A*求解每个盘面")
        print("  - 从成功的解法轨迹中学习价值函数")
        print()
        
        num_games = int(input("训练盘面数量 (推荐30-100): ") or "50")
        timeout = float(input("每个盘面A*求解时限/秒 (推荐20-40): ") or "30")
        lr = float(input("学习率 (推荐0.001): ") or "0.001")
        
        print(f"\n开始训练...")
        print(f"预计最长时间: {num_games * timeout / 60:.1f} 分钟\n")
        
        trainer = RLTrainer(learning_rate=lr)
        success = trainer.train_from_expert_trajectories(
            num_games=num_games,
            timeout_per_game=timeout,
            verbose=True
        )
        
        if success:
            # 保存模型
            trainer.value_network.save("shenzhen_rl_model_v6.npz")
            
            # 询问是否测试
            test = input("\n是否测试训练后的模型？(y/n): ").strip().lower()
            if test == 'y':
                print("\n" + "="*70)
                print("测试训练后的模型")
                print("="*70)
                
                test_status = create_random_layout()
                print("\n测试盘面:")
                test_status.print_status()
                
                print("使用RL增强求解器...")
                solver = RLEnhancedSolver(test_status, trainer, rl_weight=0.3)
                solution = solver.solve(timeout=60, verbose=True)
                
                if solution:
                    print(f"\n✓ 找到解法！")
                    print(f"  步数: {len(solution)}")
                    print(f"  探索节点: {solver.nodes_explored}")
                else:
                    print(f"\n✗ 未在60秒内找到解法")
        else:
            print("\n✗ 训练失败")
            print("建议：增加求解时限或减少训练盘面数量")
    
    elif choice == "2":
        # 使用RL模型求解
        print("\n" + "="*70)
        print("模式2：使用RL模型求解")
        print("="*70)
        
        try:
            trainer = RLTrainer()
            trainer.value_network.load("shenzhen_rl_model_v6.npz")
            print()
        except:
            print("\n✗ 未找到模型文件 shenzhen_rl_model_v6.npz")
            print("请先运行模式1训练模型")
            return
        
        # 生成盘面
        seed_input = input("\n输入随机种子（留空则完全随机）: ").strip()
        if seed_input:
            seed = int(seed_input)
            status = create_random_layout(seed)
        else:
            status = create_random_layout()
        
        print("\n初始盘面:")
        status.print_status()
        
        # 显示价值评估
        value = trainer.evaluate_value(status)
        print(f"RL价值评估: {value:.4f} (更负 = 更远离目标)\n")
        
        # RL权重
        rl_weight = float(input("RL启发式权重 0-1 (推荐0.3): ") or "0.3")
        
        print(f"\n开始求解 (RL权重={rl_weight})...")
        start_time = time.time()
        solver = RLEnhancedSolver(status, trainer, rl_weight=rl_weight)
        solution = solver.solve(timeout=180, verbose=True)
        elapsed = time.time() - start_time
        
        if solution:
            print(f"\n{'='*70}")
            print("✓ 找到解法！")
            print(f"{'='*70}")
            print(f"  步数: {len(solution)}")
            print(f"  探索节点: {solver.nodes_explored}")
            print(f"  用时: {elapsed:.2f} 秒")
            print(f"  平均每步探索: {solver.nodes_explored/len(solution):.1f} 节点")
            
            show = input("\n显示详细步骤？(y/n): ").strip().lower()
            if show == 'y':
                print(f"\n{'='*70}")
                print("解法步骤:")
                print(f"{'='*70}")
                for i, step in enumerate(solution, 1):
                    print(f"  {i:3d}. {step}")
        else:
            print(f"\n✗ 未在180秒内找到解法")
            print(f"  探索了 {solver.nodes_explored} 个节点")
    
    elif choice == "3":
        # 传统A*求解
        print("\n" + "="*70)
        print("模式3：传统A*求解")
        print("="*70)
        
        seed_input = input("\n输入随机种子（留空则完全随机）: ").strip()
        if seed_input:
            status = create_random_layout(int(seed_input))
        else:
            status = create_random_layout()
        
        print("\n初始盘面:")
        status.print_status()
        
        print("开始求解...")
        solver = RLEnhancedSolver(status, trainer=None)
        solution = solver.solve(timeout=180, verbose=True)
        
        if solution:
            elapsed_time = time.time()
            print(f"\n✓ 找到解法！")
            print(f"  步数: {len(solution)}")
            print(f"  探索节点: {solver.nodes_explored}")
        else:
            print(f"\n✗ 未在180秒内找到解法")
    
    elif choice == "4":
        # 对比测试
        print("\n" + "="*70)
        print("模式4：对比测试")
        print("="*70)
        
        try:
            trainer = RLTrainer()
            trainer.value_network.load("shenzhen_rl_model_v6.npz")
            print()
        except:
            print("\n✗ 未找到模型，将仅测试传统方法")
            trainer = None
        
        num_tests = int(input("测试盘面数量 (推荐5-10): ") or "5")
        timeout = float(input("每个盘面求解时限/秒 (推荐30-60): ") or "60")
        
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
            
            # 生成盘面
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
                solver_rl = RLEnhancedSolver(status.copy(), trainer, rl_weight=0.3)
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
                avg_steps = sum(data['steps']) / solved
                avg_nodes = sum(data['nodes']) / solved
                avg_time = sum(data['times']) / solved
                print(f"  平均步数: {avg_steps:.1f}")
                print(f"  平均探索节点: {avg_nodes:.0f}")
                print(f"  平均用时: {avg_time:.2f}秒")
        
        print_stats("传统A*", results['traditional'])
        
        if trainer:
            print_stats("RL增强", results['rl'])
            
            # 改进分析
            if results['traditional']['solved'] > 0 and results['rl']['solved'] > 0:
                avg_steps_trad = sum(results['traditional']['steps']) / results['traditional']['solved']
                avg_steps_rl = sum(results['rl']['steps']) / results['rl']['solved']
                avg_nodes_trad = sum(results['traditional']['nodes']) / results['traditional']['solved']
                avg_nodes_rl = sum(results['rl']['nodes']) / results['rl']['solved']
                avg_time_trad = sum(results['traditional']['times']) / results['traditional']['solved']
                avg_time_rl = sum(results['rl']['times']) / results['rl']['solved']
                
                print(f"\n改进分析:")
                step_diff = avg_steps_rl - avg_steps_trad
                node_diff = avg_nodes_rl - avg_nodes_trad
                time_diff = avg_time_rl - avg_time_trad
                
                print(f"  步数: {step_diff:+.1f} ({100*step_diff/avg_steps_trad:+.1f}%)")
                print(f"  节点: {node_diff:+.0f} ({100*node_diff/avg_nodes_trad:+.1f}%)")
                print(f"  时间: {time_diff:+.2f}s ({100*time_diff/avg_time_trad:+.1f}%)")
    
    elif choice == "5":
        # 快速演示
        print("\n" + "="*70)
        print("模式5：快速演示")
        print("="*70)
        print("\n说明：")
        print("  - 快速训练（少量盘面）")
        print("  - 自动测试训练后的模型")
        print("  - 适合快速体验RL功能")
        print()
        
        # 使用较小的训练集
        num_games = 10
        timeout = 20
        
        print(f"开始快速训练（{num_games}个盘面，时限{timeout}秒/盘面）...")
        print()
        
        trainer = RLTrainer(learning_rate=0.001)
        success = trainer.train_from_expert_trajectories(
            num_games=num_games,
            timeout_per_game=timeout,
            verbose=True
        )
        
        if success:
            # 自动测试
            print("\n" + "="*70)
            print("自动测试训练后的模型")
            print("="*70)
            
            # 测试3个盘面
            num_tests = 3
            test_timeout = 30
            
            for test_num in range(1, num_tests + 1):
                print(f"\n测试 {test_num}/{num_tests}")
                print("-"*70)
                
                test_status = create_random_layout(seed=3000 + test_num)
                print("盘面预览:")
                test_status.print_status(compact=True)
                
                # 传统方法
                print("\n  传统A*求解...", end=' ', flush=True)
                solver_trad = RLEnhancedSolver(test_status.copy(), trainer=None)
                sol_trad = solver_trad.solve(timeout=test_timeout, verbose=False)
                
                if sol_trad:
                    print(f"✓ (步数:{len(sol_trad)}, 节点:{solver_trad.nodes_explored})")
                else:
                    print("✗ 超时")
                
                # RL方法
                print("  RL增强求解...", end=' ', flush=True)
                solver_rl = RLEnhancedSolver(test_status.copy(), trainer, rl_weight=0.3)
                sol_rl = solver_rl.solve(timeout=test_timeout, verbose=False)
                
                if sol_rl:
                    print(f"✓ (步数:{len(sol_rl)}, 节点:{solver_rl.nodes_explored})")
                else:
                    print("✗ 超时")
                
                # 比较
                if sol_trad and sol_rl:
                    node_improvement = solver_trad.nodes_explored - solver_rl.nodes_explored
                    if node_improvement > 0:
                        print(f"  → RL方法减少了 {node_improvement} 个节点探索 ({100*node_improvement/solver_trad.nodes_explored:.1f}%)")
                    elif node_improvement < 0:
                        print(f"  → RL方法增加了 {-node_improvement} 个节点探索")
                    else:
                        print(f"  → 两种方法探索节点数相同")
            
            print("\n" + "="*70)
            print("快速演示完成！")
            print("="*70)
            print("\n提示：")
            print("  - 更多训练数据可以提升RL性能")
            print("  - 可以选择模式1进行完整训练")
            print("  - 训练后的模型可以用模式2求解具体盘面")
        else:
            print("\n✗ 快速训练失败")
            print("建议：尝试其他模式")
    
    else:
        print("\n✗ 无效的选择，请输入1-5")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已中断")
    except Exception as e:
        print(f"\n\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
