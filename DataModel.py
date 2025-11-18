
from typing import List, Optional, Set, Iterator
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

    def __repr__(self) -> str:
        if self.type == CardType.NUMBER:
            return f"N({self.color},{self.num})"
        elif self.type == CardType.FLOWER:
            return f"F({self.color},{self.value})"
        else:
            return f"S({self.value})"

    def __hash__(self) -> int:
        # stash 使用 set，需要 hashable
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return False
        return self.value == other.value


class Stack:
    """
    牌堆（列）
    cards: bottom ... top (length -1 为顶)
    continuous_length: 从顶往下连续可作为一组移动的长度（>=0）
    """
    def __init__(self, cards_list: Optional[List[Card]] = None, continuous_length: int = 0, stack: Optional['Stack'] = None):
        if stack:
            self.cards = [Card(c.value) for c in stack.cards]
            self.continuous_length = stack.continuous_length
        elif cards_list:
            self.cards = cards_list
            if continuous_length:
                self.continuous_length = continuous_length
            else:
                self.recompute_continuous_length()
        else:
            self.cards: List[Card] = list()
            self.continuous_length: int = 0

    def recompute_continuous_length(self) -> None:
        # 计算从顶开始的连续序列长度（规则：数字牌，颜色交替，数字依次递减1）
        n = len(self.cards)
        if n == 0:
            self.continuous_length = 0
            return
        # 最少 1（顶张自己）
        length = 1
        for i in range(n - 1, 0, -1):
            cur = self.cards[i]
            prev = self.cards[i - 1]
            if cur.type == CardType.NUMBER and prev.type == CardType.NUMBER and cur.color != prev.color and cur.num == prev.num - 1:
                length += 1
            else:
                break
        self.continuous_length = length
    
    def move_group(self, count: int) -> List[Card]:
        """
        从栈顶移动 count 张（保证 count <= continuous_length）
        返回的列表顺序是从底到顶（即被移动组在目标堆 push 时，逐个 push 可以保持原顺序）
        """
        if count <= 0:
            return []
        if count > len(self.cards):
            raise IndexError("not enough cards to move")
        if count > self.continuous_length:
            raise ValueError(f"cannot move {count}, continuous_length={self.continuous_length}")
        start = len(self.cards) - count
        group = self.cards[start:]
        self.cards = self.cards[:start]
        self.continuous_length = max(0, self.continuous_length - count)
        if self.continuous_length == 0:
            self.recompute_continuous_length()
        return group

        
    def pop(self) -> Card:
        if not self.cards:
            raise IndexError("pop from empty stack")
        card = self.cards.pop()
        self.continuous_length = max(0, self.continuous_length - 1)
        if self.continuous_length == 0:
            self.recompute_continuous_length()
        return card

    def top(self) -> Optional[Card]:
        return self.cards[-1] if self.cards else None

    def push(self, card: Card) -> None:
        self.cards.append(card)
        self.recompute_continuous_length() 
    
    def __repr__(self) -> str:
        return f"Stack({self.cards}, cont={self.continuous_length})"

    def __deepcopy__(self, memo):
        new_stack = Stack(cards_list=[Card(c.value) for c in self.cards])
        new_stack.continuous_length = self.continuous_length
        return new_stack

class Stash:
    def __init__(self):
        self.cards: Set[Card] = set()
        self.limit: int = 3

    def can_add(self) -> bool:
        return len(self.cards) < self.limit

    def add(self, card: Card) -> None:
        if not self.can_add():
            raise ValueError("Stash is full")
        self.cards.add(card)

    def reduce_limit(self) -> None:
        self.limit = max(0, self.limit - 1)

    def remove(self, card: Card) -> Card:
        """按 value 删除对应卡"""
        for c in list(self.cards):
            if c == card:
                self.cards.remove(c)
                return c
        raise ValueError("Card not found in stash")

    def find_by_color(self, color: int) -> List[Card]:
        return [c for c in self.cards if c.color == color]

    def __iter__(self) -> Iterator[Card]:
        """支持 for card in self.stash 迭代"""
        return iter(self.cards)

    def __repr__(self) -> str:
        return f"Stash({self.cards})"

    def __deepcopy__(self, memo):
        new_stash = Stash()
        new_stash.cards = {Card(c.value) for c in self.cards}
        new_stash.limit = self.limit
        return new_stash

class SortedArea:
    """按颜色存储清除过的牌"""
    def __init__(self):
        self.stacks: List[Stack] = [Stack() for _ in range(3)]

    def push(self, card: Card) -> None:
        self.stacks[card.color].push(card)

    def top(self, color: int) -> Optional[Card]:
        return self.stacks[color].top()

    def __repr__(self) -> str:
        return f"Sorted({self.stacks})"

    def __deepcopy__(self, memo):
        new_sorted = SortedArea()
        new_sorted.stacks = [s.__deepcopy__(memo) for s in self.stacks]
        return new_sorted


class Status:
    def __init__(self):
        self.stacks: List[Stack] = [Stack() for _ in range(8)]
        self.stash: Stash = Stash()
        self.sorted: SortedArea = SortedArea()
        self.special_card_removed: bool = False

    def __deepcopy__(self, memo):
        new_status = Status()
        new_status.stacks = [s.__deepcopy__(memo) for s in self.stacks]
        new_status.stash = self.stash.__deepcopy__(memo)
        new_status.sorted = self.sorted.__deepcopy__(memo)
        new_status.special_card_removed = self.special_card_removed
        return new_status

    def move(self, from_area: Area, from_idx: int, to_area: Area, to_idx: Optional[int] = None, count: int = 1) -> None:
        """统一的移动函数"""

        # --- 取牌 ---
        if from_area == Area.STACK:
            cards = self.stacks[from_idx].move(count)
        elif from_area == Area.STASH:
            if count != 1:
                raise ValueError("Stash move count must be 1")
            cards = [self.stash.remove(from_idx)]
        else:
            raise ValueError("Cannot move out of sorted")

        # --- 放牌 ---
        if to_area == Area.STACK:
            for c in cards:
                self.stacks[to_idx].append(c)

        elif to_area == Area.STASH:
            if len(cards) != 1:
                raise ValueError("Can only move 1 card into stash")
            self.stash.add(cards[0])

        else:
            raise ValueError("Cannot move into sorted")
        self.auto_remove()

    # ========== 状态转移：pop ==========
    def pop(self, area: Area, idx: int = None, card: Card = None) -> Card:
        if area == Area.STACK:
            removed = self.stacks[idx].pop()

        elif area == Area.STASH:
            removed = self.stash.remove(card)

        else:
            raise ValueError("Cannot pop from SORTED area")

        return removed

    def _can_auto_remove_card(self, card: Card) -> bool:
        """数字牌和特殊牌自动归位判断"""

        if card.type == CardType.SPECIAL:
            return True

        if card.type == CardType.NUMBER:
            top = self.sorted.top(card.color)
            if not top:
                return card.num == 1

            if card.num == top.num + 1:
                # 如果已排序区域的顶为1，则同色的2不需要考虑其他1的依赖。
                if top.num == 1:
                    return True
                else:
                    sorted_color_1 = self.sorted.top((card.color+1)%3)
                    sorted_color_2 = self.sorted.top((card.color+2)%3)
                    if sorted_color_1 and sorted_color_2:
                        if min(sorted_color_1.num ,sorted_color_2.num) >= card.num:
                            return True
        return False

    def _auto_remove_flowers(self) -> bool:
        """花牌 4 张同色自动移除"""

        flower_by_color = {0: [], 1: [], 2: []}

        # stack 顶部
        for idx, st in enumerate(self.stacks):
            t = st.top()
            if t and t.type == CardType.FLOWER:
                flower_by_color[t.color].append((Area.STACK, idx, t))

        # stash
        for c in self.stash.cards:
            if c.type == CardType.FLOWER:
                flower_by_color[c.color].append((Area.STASH, c.value, c))

        # check
        for color, items in flower_by_color.items():
            if len(items) == 4:
                # --- 检查是否满足移除条件 ---
                # 原版逻辑是：
                # 1. 如果 (len(status.stash) < 3 - status.used_stash)
                # 2. 或者，如果 Stash 中已存在该 color 的牌
                # 满足任一条件即可移除。
                
                has_free_slot_condition = self.stash.can_add()
                has_card_in_stash = any([card.color == color for card in self.stash if card.type == CardType.FLOWER])
                if has_free_slot_condition or has_card_in_stash:
                    for area, id_, card in items:
                        if area == Area.STACK:
                            self.stacks[id_].pop()
                        else:
                            self.stash.remove(card)
                    self.stash.reduce_limit()
                    return True
        return False

    def auto_remove(self) -> None:
        """自动移除所有可以移除的牌"""
        changed = True
        while changed:
            changed = False

            # 1. 检查并移除Stask和Stash中的牌
            for idx, st in enumerate(self.stacks):
                top = st.top()
                if top and self._can_auto_remove_card(top):
                    card = self.pop(Area.STACK, idx=idx)
                    if card.type == CardType.SPECIAL:
                        self.special_card_removed = True
                    else:
                        self.sorted.push(card)
                    changed = True
                    break # 状态已变，从while开头重新检查
            if changed:
                continue

            for card in list(self.stash):
                if self._can_auto_remove_card(card):
                    card = self.pop(Area.STASH, card=card)
                    if card.type == CardType.SPECIAL:
                        self.special_card_removed = True
                    else:
                        self.sorted.push(card)
                    changed = True
                    break # 状态已变，从while开头重新检查
            if changed:
                continue

            # 2. 检查并移除花牌
            if self._auto_remove_flowers():
                changed = True
                continue
