# -*- coding: utf-8 -*-
import heapq
import time
import hashlib
import random
from typing import List, Optional, Tuple, Any
from enum import Enum
from collections import namedtuple

class CardType(Enum):
    NUMBER = 0
    FLOWER = 1
    SPECIAL = 2

class Area(Enum):
    STACK = 0
    STASH = 1

class Card:
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
            color_names = ['R', 'G', 'B']  # 红、绿、蓝
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
    def __init__(self):
        self.cards: List[Card] = []
        self.cont_len = 0   # 从顶开始的连续长度

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
    def __init__(self):
        self.stacks: List[Stack] = [Stack() for _ in range(8)]
        self.stash: set[Card] = set()
        self.stash_limit = 3
        self.sorted_tops = [0, 0, 0]      # 每色已收到的最大数字
        self.special_removed = False

    def print_status(self):
        """打印当前状态：先打印stash和已收牌（横向），再竖向逐列打印Stack"""
        # 打印 stash
        stash_cards = sorted(list(self.stash), key=lambda c: c.value)
        stash_str = " ".join(repr(c) for c in stash_cards) if stash_cards else "空"
        print(f"Stash({len(stash_cards)}/{self.stash_limit}): {stash_str}")
        
        # 打印已收的牌
        collected_str = f"红:{self.sorted_tops[0]} 绿:{self.sorted_tops[1]} 蓝:{self.sorted_tops[2]}"
        special_str = "已收" if self.special_removed else "未收"
        print(f"已收: {collected_str}  特殊牌:{special_str}")
        
        # 打印列头
        print("     " + "".join(f"列{i+1:2}        " for i in range(8)))
        
        # 竖向逐列打印 Stack
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

    # ---------- 状态哈希 ----------
    def hash_key(self) -> bytes:
        parts = []
        MAX_LEN = 20
        for st in self.stacks:
            col = [c.value for c in st.cards] + [255] * (MAX_LEN - len(st.cards))
            parts.extend(col[:MAX_LEN])
        stash_vals = sorted(c.value for c in self.stash)
        stash_vals += [255] * (3 - len(stash_vals))
        parts.extend(stash_vals)
        parts.extend(self.sorted_tops)
        parts.append(3 - self.stash_limit)
        parts.append(1 if self.special_removed else 0)
        return hashlib.md5(bytes(parts)).digest()

    # ---------- 自动消除 ----------
    def _can_auto_remove(self, card: Card) -> bool:
        if card.type == CardType.SPECIAL:
            return True
        if card.type != CardType.NUMBER:
            return False
        need = card.num
        color = card.color
        top = self.sorted_tops[color]
        if top + 1 == need:
            if top == 0 or top >= 1 and min(self.sorted_tops[(color+1)%3], self.sorted_tops[(color+2)%3]) >= top:
                return True
        return False

    def _auto_remove_flowers(self) -> bool:
        count = [0, 0, 0]
        sources = []  # (is_stack, idx_or_card)
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
                # 判断是否允许消除（stash有空位 或 stash里已有同色花牌）
                has_slot = len(self.stash) < self.stash_limit
                has_same_in_stash = any(c.type == CardType.FLOWER and c.color == color for c in self.stash)
                if has_slot or has_same_in_stash:
                    # 执行消除
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
            # 数字牌 & 特殊牌
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
            # 花牌
            if self._auto_remove_flowers():
                changed = True

    # ---------- 胜利判断 ----------
    def is_solved(self) -> bool:
        if not self.special_removed:
            return False
        for st in self.stacks:
            if st.cards:
                return False
        for c in self.stash:
            if c.type != CardType.FLOWER:  # 花牌已经被 auto_remove_flowers 清除
                return False
        return True


# ---------- undo 需要的 Move ----------
Move = namedtuple("Move", ["type", "src", "dst", "cards", "cnt"])

# 把原来的 Solver 类整个替换成下面这个
class Solver:
    def __init__(self, status: Status):
        self.start = status

    def solve(self, timeout=180) -> Optional[List[str]]:
        start_time = time.time()
        visited = set()
        heap = []
        counter = 0

        start_copy = self._fast_copy(self.start)
        h = self._heuristic(start_copy)
        heapq.heappush(heap, (h, 0, counter, start_copy, []))
        visited.add(start_copy.hash_key())

        while heap:
            if time.time() - start_time > timeout:
                print("超时")
                return None

            _, g, _, curr, path = heapq.heappop(heap)

            if curr.is_solved():
                print(f"找到解法！步数 {len(path)}，用时 {time.time()-start_time:.2f}s")
                return path

            for desc, next_status in self._successors(curr):
                key = next_status.hash_key()
                if key in visited:
                    continue
                visited.add(key)

                new_h = self._heuristic(next_status)
                heapq.heappush(heap, (g + 1 + new_h, g + 1, counter := counter + 1, next_status, path + [desc]))

        print("未找到解")
        return None

    def _fast_copy(self, s: Status) -> Status:
        new_s = Status()
        for i, st in enumerate(s.stacks):
            for c in st.cards:
                new_s.stacks[i].push(Card(c.value))  # Card 本身轻量
            new_s.stacks[i].cont_len = st.cont_len
        new_s.stash = {Card(c.value) for c in s.stash}
        new_s.stash_limit = s.stash_limit
        new_s.sorted_tops = s.sorted_tops[:]
        new_s.special_removed = s.special_removed
        return new_s

    def _heuristic(self, s: Status) -> int:
        # 和原来一样
        h = 0
        collected = sum(s.sorted_tops)
        h += (27 - collected) * 4
        if not s.special_removed: h += 25
        flower_cnt = [0]*3
        for st in s.stacks:
            if st.top() and st.top().type == CardType.FLOWER:
                flower_cnt[st.top().color] += 1
        for c in s.stash:
            if c.type == CardType.FLOWER:
                flower_cnt[c.color] += 1
        for c in flower_cnt:
            h += max(0, 4 - c) * 5
        empty = sum(1 for st in s.stacks if not st.cards)
        h -= empty * 10
        h += len(s.stash) * 6
        for st in s.stacks:
            h -= (st.cont_len - 1) * 2
        return max(0, h)

    def _successors(self, s: Status):
        base = self._fast_copy(s)   # 只拷贝一次

        # stack → stack
        for src_i in range(8):
            src_base = base.stacks[src_i]
            if not src_base.cards:
                continue
            movable = src_base.movable_count()
            for cnt in range(1, movable + 1):
                group = src_base.cards[-cnt:]
                bottom = group[0]

                for dst_i in range(8):
                    if src_i == dst_i:
                        continue
                    # 每次尝试都从干净的 base 开始
                    cur = self._fast_copy(base)
                    src = cur.stacks[src_i]
                    dst = cur.stacks[dst_i]

                    can = not dst.cards or \
                        (bottom.type == CardType.NUMBER and
                        dst.top().type == CardType.NUMBER and
                        bottom.color != dst.top().color and
                        bottom.num == dst.top().num - 1)

                    if can:
                        moved = [src.pop() for _ in range(cnt)][::-1]
                        for c in moved:
                            dst.push(c)
                        cur.auto_remove()
                        yield f"列{src_i+1} → 列{dst_i+1} ({cnt}张)", cur

        # stack → stash
        if len(base.stash) < base.stash_limit:
            for i in range(8):
                if base.stacks[i].cards:
                    cur = self._fast_copy(base)
                    card = cur.stacks[i].pop()
                    cur.stash.add(card)
                    cur.auto_remove()
                    yield f"列{i+1} → Stash ({card})", cur

        # stash → stack
        for card in list(base.stash):
            for i in range(8):
                cur = self._fast_copy(base)
                dst = cur.stacks[i]
                if not dst.cards or \
                (card.type == CardType.NUMBER and dst.top().type == CardType.NUMBER and
                    card.color != dst.top().color and card.num == dst.top().num - 1):
                    cur.stash.remove(card)
                    dst.push(card)
                    cur.auto_remove()
                    yield f"Stash ({card}) → 列{i+1}", cur
# ==================== 发牌 & 测试 ====================
def create_random_layout() -> Status:
    s = Status()
    cards = [Card(i) for i in range(40)]
    random.shuffle(cards)
    for i in range(8):
        for j in range(5):
            s.stacks[i].push(cards[i*5 + j])
    s.auto_remove()
    return s


def simulate_solution(initial_status: Status, solution: List[str]):
    """模拟每一步操作并打印盘面"""
    import re
    
    # 深度复制初始状态
    curr = Solver(initial_status)._fast_copy(initial_status)
    
    print("=== 初始状态 ===")
    curr.print_status()
    
    for step_num, step_desc in enumerate(solution, 1):
        print(f"{'='*80}")
        print(f"第 {step_num} 步: {step_desc}")
        print(f"{'='*80}")
        
        # 解析步骤描述并执行操作
        # 列3 → 列2 (1张)
        match = re.match(r"列(\d+) → 列(\d+) \((\d+)张\)", step_desc)
        if match:
            src_i, dst_i, cnt = int(match.group(1)) - 1, int(match.group(2)) - 1, int(match.group(3))
            moved = [curr.stacks[src_i].pop() for _ in range(cnt)][::-1]
            for c in moved:
                curr.stacks[dst_i].push(c)
            curr.auto_remove()
            curr.print_status()
            continue
        
        # 列5 → Stash (2G)
        match = re.match(r"列(\d+) → Stash \((.+)\)", step_desc)
        if match:
            src_i = int(match.group(1)) - 1
            card = curr.stacks[src_i].pop()
            curr.stash.add(card)
            curr.auto_remove()
            curr.print_status()
            continue
        
        # Stash (2G) → 列6
        match = re.match(r"Stash \((.+)\) → 列(\d+)", step_desc)
        if match:
            card_str, dst_i = match.group(1), int(match.group(2)) - 1
            # 从stash中找到匹配的卡片
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


if __name__ == "__main__":
    random.seed()   # 固定种子方便调试，想随机去掉这行即可

    status = create_random_layout()

    print("=== 初始牌局 ===")
    status.print_status()

    solver = Solver(status)
    solution = solver.solve(timeout=180)

    if solution:
        print(f"\n找到解法！共 {len(solution)} 步")
        print(f"\n{'='*80}")
        print("开始模拟求解过程...")
        print(f"{'='*80}\n")
        simulate_solution(status, solution)
    else:
        print("\n本次随机局未在 3 分钟内解出（极难局存在，换个种子再试）")
