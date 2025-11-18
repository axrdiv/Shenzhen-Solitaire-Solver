# -*- coding: utf-8 -*-
"""
Shenzhen Solitaire 优化求解器（单文件）
包含多项优化：
 - 更快的 _fast_copy（列表拷贝而非逐卡 push）
 - 状态 canonical hash（对称性消除）
 - last_move 记录与优先级（优先扩展与上一步相关的列/stash）
 - 更强启发式：depth penalty, cont_len^2 bonus, stash/empty 权重调整
 - move ordering：生成所有候选后按优先级排序再扩展
 - 一些简单的无意义动作剪枝（避免无用的 stack->stash / 单卡到空列）

注：为了可读性保留原来的大部分结构并在关键点优化。
"""

import heapq
import time
import hashlib
import random
from typing import List, Optional, Tuple
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
        # 记录上一步移动用于启发（-1 表示无）
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

    # ---------- 状态哈希（canonical）----------
    def hash_key(self) -> bytes:
        # canonicalize stacks by their tuple representation to break symmetry
        stacks_repr = [tuple(c.value for c in st.cards) for st in self.stacks]
        # 排序后的堆表示（但需要保持 stash & sorted_tops，因为它们区别化状态）
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
        # last move not included in hash to avoid over-fragmentation
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
        # 这个函数尽可能高效：先快速检查是否存在可消除的顶
        while changed:
            changed = False
            # 检查 stack 顶
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
            # 检查 stash
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

    # ---------- 胜利判断 ----------
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


# ---------- undo 需要的 Move ----------
Move = namedtuple("Move", ["type", "src", "dst", "cards", "cnt"])


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

            # 生成后继并按优先级排序
            succs = list(self._successors(curr))
            # _successors 已按优先级 yield (priority, desc, status)
            for priority, desc, next_status in succs:
                key = next_status.hash_key()
                if key in visited:
                    continue
                visited.add(key)
                new_h = self._heuristic(next_status)
                heapq.heappush(heap, (g + 1 + new_h - priority/100.0, g + 1, counter := counter + 1, next_status, path + [desc]))

        print("未找到解")
        return None

    def _fast_copy(self, s: Status) -> Status:
        new_s = Status()
        # 列表复制（比 push 快）
        for i, st in enumerate(s.stacks):
            # shallow copy of card objects (Card is immutable-ish)
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
        # Improved heuristic combining multiple signals
        h = 0
        collected = sum(s.sorted_tops)
        # 未收的数字牌估计成每张 4 步
        h += (27 - collected) * 4
        # special penalty
        if not s.special_removed:
            h += 20
        # flower shortfall per color
        flower_cnt = [0]*3
        for st in s.stacks:
            if st.top() and st.top().type == CardType.FLOWER:
                flower_cnt[st.top().color] += 1
        for c in s.stash:
            if c.type == CardType.FLOWER:
                flower_cnt[c.color] += 1
        for c in flower_cnt:
            h += max(0, 4 - c) * 4
        # stash penalty
        h += len(s.stash) * 8
        # empty columns are valuable
        empty = sum(1 for st in s.stacks if not st.cards)
        h -= empty * 12
        # reward long contiguous chains (square bonus)
        for st in s.stacks:
            h -= (st.cont_len * st.cont_len)
        # depth penalty: 若关键牌被埋很深，加重惩罚
        # 对每种颜色，找下一个需要的牌的深度
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
            # stash 优先级视为深度较小
            for c in s.stash:
                if c.type == CardType.NUMBER and c.color == color and c.num == need:
                    depth = min(depth, 1)
            if depth < 99:
                h += depth * 10
            else:
                h += 30  # 没找到下一张，给较大惩罚
        return max(0, int(h))

    def _successors(self, s: Status):
        base = self._fast_copy(s)
        candidates: List[Tuple[int, str, Status]] = []  # (priority, desc, status)

        # helper: evaluate status change to build priority
        def evaluate_and_append(desc: str, cur: Status, moved_from_stash=False, src_i=-1, dst_i=-1, moved_cnt=0):
            # compute priority: prefer moves that trigger auto_remove, clear a stack, involve last move
            pr = 0
            # triggered removals? compare sorted_tops
            delta_collected = sum(cur.sorted_tops) - sum(s.sorted_tops)
            if cur.special_removed and not s.special_removed:
                pr += 80
            pr += delta_collected * 10
            # cleared a column?
            cleared = 0
            if src_i >= 0 and len(cur.stacks[src_i].cards) == 0:
                pr += 30
                cleared = 1
            # if dst becomes longer cont_len
            if dst_i >= 0:
                pr += cur.stacks[dst_i].cont_len * 2
            # last move relevance
            if src_i == s.last_src or dst_i == s.last_dst:
                pr += 8
            if moved_from_stash and s.last_was_stash:
                pr += 10
            # smaller stash preferred
            pr += (s.stash_limit - len(cur.stash)) * 3
            # discourage extra long stash usage
            pr -= len(cur.stash) * 2
            candidates.append((pr, desc, cur))

        # ---------- stack -> stack ----------
        for src_i in range(8):
            src_base = base.stacks[src_i]
            if not src_base.cards:
                continue
            movable = src_base.movable_count()
            # try different counts (1..movable)
            for cnt in range(1, movable + 1):
                group_bottom = src_base.cards[-cnt]
                # prune: 不要把单张随意放到空列（除非它能清空 src）
                for dst_i in range(8):
                    if src_i == dst_i:
                        continue
                    dst_base = base.stacks[dst_i]
                    can = (not dst_base.cards) or (
                        group_bottom.type == CardType.NUMBER and dst_base.top() and dst_base.top().type == CardType.NUMBER and
                        group_bottom.color != dst_base.top().color and group_bottom.num == dst_base.top().num - 1)
                    if not can:
                        continue
                    # pruning rule: single card to empty column is usually bad
                    if cnt == 1 and not dst_base.cards and len(src_base.cards) > 1 and src_base.cont_len != len(src_base.cards):
                        # 只有当移动会清空 src 或增加 cont_len 才允许
                        continue
                    # make copy, apply move and auto_remove
                    cur = self._fast_copy(base)
                    src = cur.stacks[src_i]
                    dst = cur.stacks[dst_i]
                    moved = [src.pop() for _ in range(cnt)][::-1]
                    for c in moved:
                        dst.push(c)
                    # record last move
                    cur.last_src = src_i
                    cur.last_dst = dst_i
                    cur.last_was_stash = False
                    cur.auto_remove()
                    desc = f"列{src_i+1} → 列{dst_i+1} ({cnt}张)"
                    evaluate_and_append(desc, cur, moved_from_stash=False, src_i=src_i, dst_i=dst_i, moved_cnt=cnt)

        # ---------- stack -> stash ----------
        if len(base.stash) < base.stash_limit:
            for i in range(8):
                if base.stacks[i].cards:
                    top_card = base.stacks[i].top()
                    # prune: 不要把能直接 auto_remove 的牌塞进 stash
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

        # ---------- stash -> stack ----------
        for card in list(base.stash):
            for i in range(8):
                dst_base = base.stacks[i]
                can = (not dst_base.cards) or (
                    card.type == CardType.NUMBER and dst_base.top() and dst_base.top().type == CardType.NUMBER and
                    card.color != dst_base.top().color and card.num == dst_base.top().num - 1)
                if not can:
                    continue
                cur = self._fast_copy(base)
                # find matching card instance in cur.stash
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

        # sort candidates by priority descending (higher pr -> expand first)
        candidates.sort(key=lambda x: -x[0])
        # yield as (priority, desc, status) for caller to use
        for pr, desc, cur in candidates:
            yield pr, desc, cur

    def _can_move_to_stash_prune(self, base: Status, card: Card) -> bool:
        # prune some stack->stash moves that are clearly useless
        if card is None:
            return True
        # 如果这张牌可以被 auto_remove，放入 stash 是无意义
        if base._can_auto_remove(card):
            return True
        # 数字牌：如果它小于等于已收的，没意义
        if card.type == CardType.NUMBER:
            if card.num <= base.sorted_tops[card.color]:
                return True
        # 花牌放 stash 是必要的情况比较少（通常放 stash 为了完成 4 花）
        # 但允许少数花牌进入 stash
        return False


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


if __name__ == "__main__":
    random.seed()
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
