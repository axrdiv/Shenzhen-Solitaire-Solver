# -*- coding: utf-8 -*-
"""
Shenzhen Solitaire 优化求解器（带启发函数自动优化）
新增功能：
 - 使用遗传算法自动优化启发函数权重
 - 支持在多个测试盘面上评估参数性能
 - 可以保存和加载优化后的参数
"""

import heapq
import time
import hashlib
import random
import json
from typing import List, Optional, Tuple, Dict
from enum import Enum
from collections import namedtuple
from dataclasses import dataclass, asdict

class CardType(Enum):
    NUMBER = 0
    FLOWER = 1
    SPECIAL = 2

class Area(Enum):
    STACK = 0
    STASH = 1

class Card:
    __slots__ = ("value",)
    def __init__(self, value: int):
        self.value = value

    @property
    def type(self) -> CardType:
        if self.value < 27:   return CardType.NUMBER
        if self.value < 39:   return CardType.FLOWER
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
            color_names = ['R', 'G', 'B']
            return f"{self.num}{color_names[self.color]}"
        elif self.type == CardType.FLOWER:
            return f"F{self.color+1}"
        else:
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

    def __repr__(self):
        return f"Stack({' '.join([('←' if i >= len(self.cards)-self.cont_len else '') + repr(c) for i, c in enumerate(self.cards)])})"


class Status:
    __slots__ = ("stacks", "stash", "stash_limit", "sorted_tops", "special_removed", "last_src", "last_dst", "last_was_stash")
    def __init__(self):
        self.stacks: List[Stack] = [Stack() for _ in range(8)]
        self.stash: set[Card] = set()
        self.stash_limit = 3
        self.sorted_tops = [0, 0, 0]
        self.special_removed = False
        self.last_src = -1
        self.last_dst = -1
        self.last_was_stash = False

    def print_status(self):
        stash_cards = sorted(list(self.stash), key=lambda c: c.value)
        stash_str = " ".join(repr(c) for c in stash_cards) if stash_cards else "空"
        print(f"Stash({len(stash_cards)}/{self.stash_limit}): {stash_str}")
        collected_str = f"红:{self.sorted_tops[0]} 绿:{self.sorted_tops[1]} 蓝:{self.sorted_tops[2]}"
        special_str = "已收" if self.special_removed else "未收"
        print(f"已收: {collected_str}  特殊牌:{special_str}")
        print("     " + "".join(f"列{i+1:2}        " for i in range(8)))
        max_height = max(len(st.cards) for st in self.stacks) if any(st.cards for st in self.stacks) else 0
        for row in range(max_height):
            line = f"{row+1:2}:  "
            for st in self.stacks:
                if row < len(st.cards):
                    card = st.cards[row]
                    line += f"{repr(card):11} "
                else:
                    line += "            "
            print(line)
        if max_height == 0:
            print("     (所有列为空)")
        print()

    def hash_key(self) -> bytes:
        stacks_repr = [tuple(c.value for c in st.cards) for st in self.stacks]
        sorted_stacks = sorted(stacks_repr)
        parts = []
        MAX_LEN = 20
        for col in sorted_stacks:
            col_padded = list(col) + [255] * (MAX_LEN - len(col))
            parts.extend(col_padded[:MAX_LEN])
        stash_vals = sorted(c.value for c in self.stash)
        stash_vals += [255] * (3 - len(stash_vals))
        parts.extend(stash_vals)
        parts.extend(self.sorted_tops)
        parts.append(3 - self.stash_limit)
        parts.append(1 if self.special_removed else 0)
        return hashlib.md5(bytes(parts)).digest()

    def _can_auto_remove(self, card: Card) -> bool:
        if card.type == CardType.SPECIAL:
            return True
        if card.type != CardType.NUMBER:
            return False
        need = card.num
        color = card.color
        top = self.sorted_tops[color]
        if top + 1 == need:
            if top == 0 or (top == 1 and card.num == 2) or (top >= 2 and min(self.sorted_tops[(color+1)%3], self.sorted_tops[(color+2)%3]) >= top):
                return True
        return False

    def _auto_remove_flowers(self) -> bool:
        count = [0, 0, 0]
        sources = []
        for i, st in enumerate(self.stacks):
            t = st.top()
            if t and t.type == CardType.FLOWER:
                count[t.color] += 1
                sources.append((True, i))
        for c in self.stash:
            if c.type == CardType.FLOWER:
                count[c.color] += 1
                sources.append((False, c))
        for color in range(3):
            if count[color] == 4:
                has_slot = len(self.stash) < self.stash_limit
                has_same_in_stash = any(c.type == CardType.FLOWER and c.color == color for c in self.stash)
                if has_slot or has_same_in_stash:
                    removed = 0
                    for is_stack, idx_or_card in sources:
                        if removed == 4:
                            break
                        if is_stack:
                            st = self.stacks[idx_or_card]
                            if st.top() and st.top().type == CardType.FLOWER and st.top().color == color:
                                st.pop()
                                removed += 1
                        else:
                            c = idx_or_card
                            if c.type == CardType.FLOWER and c.color == color:
                                self.stash.remove(c)
                                removed += 1
                    self.stash_limit = max(0, self.stash_limit - 1)
                    return True
        return False

    def auto_remove(self):
        changed = True
        while changed:
            changed = False
            for i, st in enumerate(self.stacks):
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


Move = namedtuple("Move", ["type", "src", "dst", "cards", "cnt"])


# ==================== 启发函数参数 ====================
@dataclass
class HeuristicParams:
    """启发函数的可调参数"""
    uncollected_weight: float = 4.0      # 未收集数字牌权重
    special_penalty: float = 20.0        # 特殊牌未收集惩罚
    flower_shortfall_weight: float = 4.0 # 花牌不足权重
    stash_penalty: float = 8.0           # stash使用惩罚
    empty_bonus: float = 12.0            # 空列奖励
    cont_len_bonus: float = 1.0          # 连续牌组奖励（平方系数）
    depth_penalty: float = 10.0          # 关键牌深度惩罚
    depth_not_found_penalty: float = 30.0 # 关键牌未找到惩罚
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'HeuristicParams':
        return cls(**d)
    
    def mutate(self, mutation_rate: float = 0.2) -> 'HeuristicParams':
        """随机变异参数"""
        params = asdict(self)
        for key in params:
            if random.random() < mutation_rate:
                # 在 ±30% 范围内变异
                factor = random.uniform(0.7, 1.3)
                params[key] = max(0.1, params[key] * factor)
        return HeuristicParams(**params)
    
    @classmethod
    def crossover(cls, p1: 'HeuristicParams', p2: 'HeuristicParams') -> 'HeuristicParams':
        """交叉两个参数集"""
        params = {}
        for key in asdict(p1):
            params[key] = getattr(p1, key) if random.random() < 0.5 else getattr(p2, key)
        return cls(**params)


class Solver:
    def __init__(self, status: Status, params: Optional[HeuristicParams] = None):
        self.start = status
        self.params = params or HeuristicParams()
        self.nodes_explored = 0  # 用于评估效率

    def solve(self, timeout=180) -> Optional[List[str]]:
        start_time = time.time()
        visited = set()
        heap = []
        counter = 0
        self.nodes_explored = 0

        start_copy = self._fast_copy(self.start)
        h = self._heuristic(start_copy)
        heapq.heappush(heap, (h, 0, counter, start_copy, []))
        visited.add(start_copy.hash_key())

        while heap:
            if time.time() - start_time > timeout:
                return None

            _, g, _, curr, path = heapq.heappop(heap)
            self.nodes_explored += 1

            if curr.is_solved():
                return path

            succs = list(self._successors(curr))
            for priority, desc, next_status in succs:
                key = next_status.hash_key()
                if key in visited:
                    continue
                visited.add(key)
                new_h = self._heuristic(next_status)
                heapq.heappush(heap, (g + 1 + new_h - priority/100.0, g + 1, counter := counter + 1, next_status, path + [desc]))

        return None

    def _fast_copy(self, s: Status) -> Status:
        new_s = Status()
        for i, st in enumerate(s.stacks):
            new_s.stacks[i].cards = st.cards.copy()
            new_s.stacks[i].cont_len = st.cont_len
        new_s.stash = {Card(c.value) for c in s.stash}
        new_s.stash_limit = s.stash_limit
        new_s.sorted_tops = s.sorted_tops[:]
        new_s.special_removed = s.special_removed
        new_s.last_src = s.last_src
        new_s.last_dst = s.last_dst
        new_s.last_was_stash = s.last_was_stash
        return new_s

    def _heuristic(self, s: Status) -> int:
        """使用可配置参数的启发函数"""
        p = self.params
        h = 0
        
        # 未收集的数字牌
        collected = sum(s.sorted_tops)
        h += (27 - collected) * p.uncollected_weight
        
        # 特殊牌惩罚
        if not s.special_removed:
            h += p.special_penalty
        
        # 花牌不足
        flower_cnt = [0]*3
        for st in s.stacks:
            if st.top() and st.top().type == CardType.FLOWER:
                flower_cnt[st.top().color] += 1
        for c in s.stash:
            if c.type == CardType.FLOWER:
                flower_cnt[c.color] += 1
        for c in flower_cnt:
            h += max(0, 4 - c) * p.flower_shortfall_weight
        
        # stash 惩罚
        h += len(s.stash) * p.stash_penalty
        
        # 空列奖励
        empty = sum(1 for st in s.stacks if not st.cards)
        h -= empty * p.empty_bonus
        
        # 连续牌组奖励
        for st in s.stacks:
            h -= (st.cont_len * st.cont_len) * p.cont_len_bonus
        
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
                h += depth * p.depth_penalty
            else:
                h += p.depth_not_found_penalty
        
        return max(0, int(h))

    def _successors(self, s: Status):
        base = self._fast_copy(s)
        candidates: List[Tuple[int, str, Status]] = []

        def evaluate_and_append(desc: str, cur: Status, moved_from_stash=False, src_i=-1, dst_i=-1, moved_cnt=0):
            pr = 0
            delta_collected = sum(cur.sorted_tops) - sum(s.sorted_tops)
            if cur.special_removed and not s.special_removed:
                pr += 80
            pr += delta_collected * 10
            if src_i >= 0 and len(cur.stacks[src_i].cards) == 0:
                pr += 30
            if dst_i >= 0:
                pr += cur.stacks[dst_i].cont_len * 2
            if src_i == s.last_src or dst_i == s.last_dst:
                pr += 8
            if moved_from_stash and s.last_was_stash:
                pr += 10
            pr += (s.stash_limit - len(cur.stash)) * 3
            pr -= len(cur.stash) * 2
            candidates.append((pr, desc, cur))

        # stack -> stack
        for src_i in range(8):
            src_base = base.stacks[src_i]
            if not src_base.cards:
                continue
            movable = src_base.movable_count()
            for cnt in range(1, movable + 1):
                group_bottom = src_base.cards[-cnt]
                for dst_i in range(8):
                    if src_i == dst_i:
                        continue
                    dst_base = base.stacks[dst_i]
                    can = (not dst_base.cards) or (
                        group_bottom.type == CardType.NUMBER and dst_base.top() and dst_base.top().type == CardType.NUMBER and
                        group_bottom.color != dst_base.top().color and group_bottom.num == dst_base.top().num - 1)
                    if not can:
                        continue
                    if cnt == 1 and not dst_base.cards and len(src_base.cards) > 1 and src_base.cont_len != len(src_base.cards):
                        continue
                    cur = self._fast_copy(base)
                    src = cur.stacks[src_i]
                    dst = cur.stacks[dst_i]
                    moved = [src.pop() for _ in range(cnt)][::-1]
                    for c in moved:
                        dst.push(c)
                    cur.last_src = src_i
                    cur.last_dst = dst_i
                    cur.last_was_stash = False
                    cur.auto_remove()
                    desc = f"列{src_i+1} → 列{dst_i+1} ({cnt}张)"
                    evaluate_and_append(desc, cur, moved_from_stash=False, src_i=src_i, dst_i=dst_i, moved_cnt=cnt)

        # stack -> stash
        if len(base.stash) < base.stash_limit:
            for i in range(8):
                if base.stacks[i].cards:
                    top_card = base.stacks[i].top()
                    if self._can_move_to_stash_prune(base, top_card):
                        continue
                    cur = self._fast_copy(base)
                    card = cur.stacks[i].pop()
                    cur.stash.add(card)
                    cur.last_src = i
                    cur.last_dst = -1
                    cur.last_was_stash = True
                    cur.auto_remove()
                    desc = f"列{i+1} → Stash ({card})"
                    evaluate_and_append(desc, cur, moved_from_stash=True, src_i=i, dst_i=-1, moved_cnt=1)

        # stash -> stack
        for card in list(base.stash):
            for i in range(8):
                dst_base = base.stacks[i]
                can = (not dst_base.cards) or (
                    card.type == CardType.NUMBER and dst_base.top() and dst_base.top().type == CardType.NUMBER and
                    card.color != dst_base.top().color and card.num == dst_base.top().num - 1)
                if not can:
                    continue
                cur = self._fast_copy(base)
                card_in_cur = None
                for c in cur.stash:
                    if c.value == card.value:
                        card_in_cur = c
                        break
                if card_in_cur is None:
                    continue
                cur.stash.remove(card_in_cur)
                cur.stacks[i].push(card_in_cur)
                cur.last_src = -1
                cur.last_dst = i
                cur.last_was_stash = True
                cur.auto_remove()
                desc = f"Stash ({card}) → 列{i+1}"
                evaluate_and_append(desc, cur, moved_from_stash=True, src_i=-1, dst_i=i, moved_cnt=1)

        candidates.sort(key=lambda x: -x[0])
        for pr, desc, cur in candidates:
            yield pr, desc, cur

    def _can_move_to_stash_prune(self, base: Status, card: Card) -> bool:
        if card is None:
            return True
        if base._can_auto_remove(card):
            return True
        if card.type == CardType.NUMBER:
            if card.num <= base.sorted_tops[card.color]:
                return True
        return False


# ==================== 遗传算法优化器 ====================
class HeuristicOptimizer:
    """使用遗传算法优化启发函数参数"""
    
    def __init__(self, population_size: int = 20, generations: int = 10):
        self.population_size = population_size
        self.generations = generations
        self.test_puzzles: List[Status] = []
    
    def generate_test_puzzles(self, count: int = 10, seed: Optional[int] = None):
        """生成测试盘面"""
        if seed is not None:
            random.seed(seed)
        
        self.test_puzzles = []
        for _ in range(count):
            s = Status()
            cards = [Card(i) for i in range(40)]
            random.shuffle(cards)
            for i in range(8):
                for j in range(5):
                    s.stacks[i].push(cards[i*5 + j])
            s.auto_remove()
            self.test_puzzles.append(s)
        
        print(f"生成了 {count} 个测试盘面")
    
    def evaluate_params(self, params: HeuristicParams, timeout: int = 30) -> float:
        """评估参数性能（返回适应度分数，越高越好）"""
        total_score = 0
        solved_count = 0
        total_steps = 0
        total_nodes = 0
        
        for puzzle in self.test_puzzles:
            solver = Solver(puzzle, params)
            solution = solver.solve(timeout=timeout)
            
            if solution:
                solved_count += 1
                total_steps += len(solution)
                total_nodes += solver.nodes_explored
                # 奖励：解决了 + 步数少 + 探索节点少
                score = 1000 - len(solution) - solver.nodes_explored / 100
            else:
                # 惩罚：未解决
                score = -100
            
            total_score += score
        
        # 综合得分
        fitness = total_score + solved_count * 500
        return fitness
    
    def optimize(self, timeout_per_puzzle: int = 30, verbose: bool = True) -> HeuristicParams:
        """运行遗传算法优化"""
        if not self.test_puzzles:
            raise ValueError("请先生成测试盘面")
        
        # 初始化种群
        population = [HeuristicParams() for _ in range(self.population_size)]
        # 添加一些随机变异的个体
        for i in range(1, self.population_size):
            population[i] = population[0].mutate(mutation_rate=0.5)
        
        best_params = None
        best_fitness = float('-inf')
        
        for gen in range(self.generations):
            if verbose:
                print(f"\n{'='*60}")
                print(f"第 {gen+1}/{self.generations} 代")
                print(f"{'='*60}")
            
            # 评估所有个体
            fitness_scores = []
            for i, params in enumerate(population):
                if verbose:
                    print(f"评估个体 {i+1}/{self.population_size}...", end=' ')
                
                fitness = self.evaluate_params(params, timeout_per_puzzle)
                fitness_scores.append((fitness, params))
                
                if verbose:
                    print(f"适应度: {fitness:.1f}")
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_params = params
            
            # 排序
            fitness_scores.sort(reverse=True, key=lambda x: x[0])
            
            if verbose:
                print(f"\n当前最佳适应度: {fitness_scores[0][0]:.1f}")
                print(f"历史最佳适应度: {best_fitness:.1f}")
            
            # 选择、交叉、变异
            new_population = []
            
            # 精英保留（保留最好的20%）
            elite_count = max(2, self.population_size // 5)
            for _, params in fitness_scores[:elite_count]:
                new_population.append(params)
            
            # 生成新个体
            while len(new_population) < self.population_size:
                # 轮盘赌选择
                parent1 = self._select_parent(fitness_scores)
                parent2 = self._select_parent(fitness_scores)
                
                # 交叉
                child = HeuristicParams.crossover(parent1, parent2)
                
                # 变异
                if random.random() < 0.3:
                    child = child.mutate(mutation_rate=0.15)
                
                new_population.append(child)
            
            population = new_population
        
        if verbose:
            print(f"\n{'='*60}")
            print("优化完成！")
            print(f"最佳适应度: {best_fitness:.1f}")
            print("\n最佳参数:")
            for key, value in best_params.to_dict().items():
                print(f"  {key}: {value:.2f}")
        
        return best_params
    
    def _select_parent(self, fitness_scores: List[Tuple[float, HeuristicParams]]) -> HeuristicParams:
        """轮盘赌选择"""
        # 将负数适应度调整为正数
        min_fitness = min(f for f, _ in fitness_scores)
        adjusted_scores = [(f - min_fitness + 1, p) for f, p in fitness_scores]
        
        total = sum(f for f, _ in adjusted_scores)
        pick = random.uniform(0, total)
        current = 0
        for fitness, params in adjusted_scores:
            current += fitness
            if current >= pick:
                return params
        return adjusted_scores[-1][1]
    
    def save_params(self, params: HeuristicParams, filename: str = "best_params.json"):
        """保存参数到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(params.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"参数已保存到 {filename}")
    
    def load_params(self, filename: str = "best_params.json") -> HeuristicParams:
        """从文件加载参数"""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return HeuristicParams.from_dict(data)


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


def simulate_solution(initial_status: Status, solution: List[str]):
    import re
    curr = Solver(initial_status)._fast_copy(initial_status)
    print("=== 初始状态 ===")
    curr.print_status()
    for step_num, step_desc in enumerate(solution, 1):
        print(f"{'='*80}")
        print(f"第 {step_num} 步: {step_desc}")
        print(f"{'='*80}")
        match = re.match(r"列(\d+) → 列(\d+) \((\d+)张\)", step_desc)
        if match:
            src_i, dst_i, cnt = int(match.group(1)) - 1, int(match.group(2)) - 1, int(match.group(3))
            moved = [curr.stacks[src_i].pop() for _ in range(cnt)][::-1]
            for c in moved:
                curr.stacks[dst_i].push(c)
            curr.auto_remove()
            curr.print_status()
            continue
        match = re.match(r"列(\d+) → Stash \((.+)\)", step_desc)
        if match:
            src_i = int(match.group(1)) - 1
            card = curr.stacks[src_i].pop()
            curr.stash.add(card)
            curr.auto_remove()
            curr.print_status()
            continue
        match = re.match(r"Stash \((.+)\) → 列(\d+)", step_desc)
        if match:
            card_str, dst_i = match.group(1), int(match.group(2)) - 1
            card_to_move = None
            for c in curr.stash:
                if repr(c) == card_str:
                    card_to_move = c
                    break
            if card_to_move:
                curr.stash.remove(card_to_move)
                curr.stacks[dst_i].push(card_to_move)
                curr.auto_remove()
            curr.print_status()
            continue
    print(f"{'='*80}")
    print("解法完成！")
    if curr.is_solved():
        print("✓ 验证成功：盘面已清空！")
    else:
        print("✗ 警告：盘面未完全清空")


# ==================== 主程序 ====================
if __name__ == "__main__":
    import sys
    
    print("Shenzhen Solitaire 优化求解器")
    print("="*60)
    print("1. 使用默认参数求解随机盘面")
    print("2. 优化启发函数参数（遗传算法）")
    print("3. 使用已保存的优化参数求解")
    print("4. 快速优化模式（少量测试盘面）")
    print("="*60)
    
    choice = input("请选择模式 (1-4): ").strip()
    
    if choice == "1":
        # 模式1：使用默认参数
        print("\n使用默认参数求解随机盘面...")
        random.seed()
        status = create_random_layout()
        print("\n=== 初始牌局 ===")
        status.print_status()
        
        solver = Solver(status)
        solution = solver.solve(timeout=180)
        
        if solution:
            print(f"\n✓ 找到解法！共 {len(solution)} 步")
            print(f"探索节点数: {solver.nodes_explored}")
            simulate_solution(status, solution)
        else:
            print("\n✗ 未在时限内找到解法")
    
    elif choice == "2":
        # 模式2：完整优化
        print("\n启动遗传算法优化器...")
        optimizer = HeuristicOptimizer(population_size=20, generations=10)
        
        test_count = int(input("生成测试盘面数量 (推荐10-20): ") or "10")
        optimizer.generate_test_puzzles(count=test_count, seed=42)
        
        timeout = int(input("每个盘面的求解时限(秒) (推荐20-40): ") or "30")
        
        print("\n开始优化（这可能需要较长时间）...")
        best_params = optimizer.optimize(timeout_per_puzzle=timeout, verbose=True)
        
        # 保存结果
        optimizer.save_params(best_params, "best_params.json")
        
        # 测试优化后的参数
        print("\n" + "="*60)
        print("使用优化后的参数测试一个新盘面...")
        test_status = create_random_layout()
        test_status.print_status()
        
        solver = Solver(test_status, best_params)
        solution = solver.solve(timeout=180)
        
        if solution:
            print(f"\n✓ 找到解法！共 {len(solution)} 步")
            print(f"探索节点数: {solver.nodes_explored}")
        else:
            print("\n✗ 未在时限内找到解法")
    
    elif choice == "3":
        # 模式3：使用已保存的参数
        print("\n加载已保存的优化参数...")
        try:
            optimizer = HeuristicOptimizer()
            params = optimizer.load_params("best_params.json")
            print("参数加载成功！")
            print("\n参数详情:")
            for key, value in params.to_dict().items():
                print(f"  {key}: {value:.2f}")
            
            print("\n使用优化参数求解随机盘面...")
            random.seed()
            status = create_random_layout()
            print("\n=== 初始牌局 ===")
            status.print_status()
            
            solver = Solver(status, params)
            solution = solver.solve(timeout=180)
            
            if solution:
                print(f"\n✓ 找到解法！共 {len(solution)} 步")
                print(f"探索节点数: {solver.nodes_explored}")
                simulate_solution(status, solution)
            else:
                print("\n✗ 未在时限内找到解法")
        
        except FileNotFoundError:
            print("错误：未找到 best_params.json 文件")
            print("请先运行模式2进行优化")
    
    elif choice == "4":
        # 模式4：快速优化
        print("\n快速优化模式（适合快速测试）...")
        optimizer = HeuristicOptimizer(population_size=10, generations=5)
        
        print("生成5个测试盘面...")
        optimizer.generate_test_puzzles(count=5, seed=42)
        
        print("\n开始快速优化...")
        best_params = optimizer.optimize(timeout_per_puzzle=20, verbose=True)
        
        optimizer.save_params(best_params, "best_params_quick.json")
        
        # 对比测试
        print("\n" + "="*60)
        print("对比测试：默认参数 vs 优化参数")
        print("="*60)
        
        test_status = create_random_layout(seed=123)
        
        # 默认参数
        print("\n使用默认参数...")
        solver1 = Solver(test_status, HeuristicParams())
        sol1 = solver1.solve(timeout=60)
        if sol1:
            print(f"✓ 步数: {len(sol1)}, 探索节点: {solver1.nodes_explored}")
        else:
            print("✗ 未解决")
        
        # 优化参数
        print("\n使用优化参数...")
        solver2 = Solver(test_status, best_params)
        sol2 = solver2.solve(timeout=60)
        if sol2:
            print(f"✓ 步数: {len(sol2)}, 探索节点: {solver2.nodes_explored}")
        else:
            print("✗ 未解决")
        
        if sol1 and sol2:
            print(f"\n改进:")
            print(f"  步数减少: {len(sol1) - len(sol2)} 步")
            print(f"  节点减少: {solver1.nodes_explored - solver2.nodes_explored} 个")
    
    else:
        print("无效选择")