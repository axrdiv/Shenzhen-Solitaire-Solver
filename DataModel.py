import hashlib
from typing import List, Optional
from enum import Enum

class CardType(Enum):
    NUMBER = 0
    FLOWER = 1
    SPECIAL = 2

class Area(Enum):
    STACK = 0
    STASH = 1

class Card:
    """
        卡片说明：
        0 ~ 26 是数字牌，x // 9 == 0,1,2 分别是三种牌色，x % 9 + 1 是牌上的数字
        27 ~ 38 是花牌，花牌没有数字只有花色。所以 (x - 27) // 4 == 0, 1, 2 是花色
        39 是特殊牌
    """
    def __init__(self, value: int = 0):
        self.value: int = value

    @property
    def type(self) -> CardType:
        if self.value < 27:
            return CardType.NUMBER
        elif self.value < 39:
            return CardType.FLOWER
        else:
            return CardType.SPECIAL
    
    @property
    def color(self) -> int:
        if self.type == CardType.NUMBER:
            return self.value // 9
        elif self.type == CardType.FLOWER:
            return (self.value - 27) // 4
        else:
            raise ValueError

    @property
    def num(self) -> int:
        if self.type == CardType.NUMBER:
            return self.value % 9 + 1
        else:
            return 0

    def __repr__(self):
        if self.type == CardType.NUMBER:
            color_names = ['R', 'G', 'B']  # 红、绿、蓝
            return f"{self.num}{color_names[self.color]}"
        elif self.type == CardType.FLOWER:
            return f"F{self.color+1}"
        else:
            return "SP"

    def __hash__(self) -> int:
        # stash 使用 set，需要 hashable
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return False
        return self.value == other.value

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
            if top == 0 or (top == 1 and card.num == 2) or (top >= 2 and min(self.sorted_tops[(color+1)%3], self.sorted_tops[(color+2)%3]) >= top):
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
