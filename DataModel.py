
from typing import List, Optional, Set
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
            return f"N({self.color},{self.num})"
        elif self.type == CardType.FLOWER:
            return f"F({self.color},{self.value})"
        else:
            return f"S({self.value})"

    def __hash__(self):
        # stash 使用 set，需要 hashable
        return hash(self.value)

    def __eq__(self, other):
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
                self.count_continuous_length()
        else:
            self.cards: List[Card] = list()
            self.continuous_length: int = 0

    def recompute_continuous_length(self):
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

    def push(self, card: Card):
        self.cards.append(card)
        self.recompute_continuous_length() 
    
    def __repr__(self):
        return f"Stack({self.cards}, cont={self.continuous_length})"

class Stash:
    def __init__(self):
        self.cards: Set[Card] = set()
        self.limit = 3

    def can_add(self) -> bool:
        return len(self.cards) < self.limit

    def add(self, card: Card):
        if not self.can_add():
            raise ValueError("Stash is full")
        self.cards.add(card)

    def reduce_limit(self):
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

    def __repr__(self):
        return f"Stash({self.cards})"



class Status:
    def __init__(self):
        self.unused_card: List[int] = [1] * 40
        self.stacks: List[Stack] = [Stack() for _ in range(8)]
        self.used_stash = 0
        self.stash: Set[Card] = Set()
        self.after_sorting: List[Stack] = list(Stack() for _ in range(3))


    def move(self):
        pass

    def pop(self, area, stack_idx: int = 0, card: Card = None):
        """从盘面中移除一张牌，会导致unused_card变化"""
        if area == Area.STACK:
            card = self.stacks[stack_idx].pop()
        else:
            if card.type == CardType.NUMBER:
                new_set = set()
                for stash_card in self.stash:
                    if stash_card.value == card.value:
                        card = stash_card
                    else:
                        new_set.add(stash_card)
                self.stash = new_set
            # CardType 是 FLOWER
            else:
                new_set = set()
                for stash_card in self.stash:
                    if stash_card.value == card.value:
                        card = stash_card
                        self.unused_card[card.value] = 0
                    else:
                        new_set.add(stash_card)
                self.stash = new_set

        return card

    
    def _check_card_can_auto_remove(self, card) -> bool:
        if card.type == CardType.SPECIAL:
            return True
        elif card.type == CardType.NUMBER:
            sorted_top = self.after_sorting[card.color].top()
            if sorted_top:
                # 如果正好到了这个数字
                if card.num == sorted_top.num + 1:
                    # 如果已排序的是数字是 1 ，则 2 号牌不需要等其他 1 的依赖
                    if sorted_top.num == 1:
                        return True
                    # 查看是否有其他依赖它的卡片，如果没有则可以自动移除这张卡片
                    if not any([self.unused_card[x * 9 + sorted_top.num] for x in range(3)]):
                        return True
            else:
                return True

    def auto_remove(self):
        for idx, st in enumerate(self.stacks):
            card = st.top()
            if self._check_card_can_auto_remove(card):
                self.after_sorting[card.color].append(card)
                self.pop(Area.STACK, stack_idx=idx)
                return self.auto_remove()

        for card in self.stash:
            if self._check_card_can_auto_remove(card):
                self.after_sorting[card.color].append(card)
                self.pop(Area.STASH, card=card)
                return self.auto_remove()



def check_flower_card(status: Status):
    """
    检查并处理桌面上的特殊牌（移除）和花色牌（凑齐4张后移除）。
    这是一个递归函数，任何状态变更（移除特殊牌或花色牌）都会触发一次新的检查。
    """
    
    # 1. 收集所有暴露的花牌位置
    #    使用列表，索引 0, 1, 2 对应牌的 color
    flower_card_locations: List[Set] = [set() for _ in range(3)]

    # 2. 检查牌堆 (Stacks)
    for idx, st in enumerate(status.stacks):
        if not st.cards:  # 跳过空牌堆
            continue
            
        top_card = st.top()

        # 规则 1: 优先移除特殊牌
        if top_card.type == CardType.SPECIAL:
            st.pop()
            # 状态已改变，必须从头重新检查
            return check_flower_card(status) 

        # 规则 2: 记录暴露的花牌
        if top_card.type == CardType.FLOWER:
            flower_card_locations[top_card.color].add((Area.STACK, idx))

    # 3. 检查 Stash 区
    for card in status.stash:
        if card.type == CardType.FLOWER:
            # 假设 card.value 是 Stash 中花牌的唯一标识
            flower_card_locations[card.color].add((Area.STASH, card.value))

    # 4. 检查是否凑齐了4张花牌
    for color, locations in enumerate(flower_card_locations):
        if len(locations) == 4:
            
            # --- 检查是否满足移除条件 ---
            # 原版逻辑是：
            # 1. 如果 (len(status.stash) < 3 - status.used_stash)
            # 2. 或者，如果 Stash 中已存在该 color 的牌
            # 满足任一条件即可移除。
            
            has_free_slot_condition = (len(status.stash) < 3 - status.used_stash)
            has_card_in_stash = any(card.color == color for card in status.stash)

            if has_free_slot_condition or has_card_in_stash:
                
                # --- 执行移除 ---
                
                # a) 从 Stacks 中移除
                for area, val in locations:
                    if area == Area.STACK:
                        status.stacks[val].pop()

                # b) 从 Stash 中移除 (使用集合推导式更简洁)
                # 检查是否需要更新 stash (仅当花牌来源包含 STASH 时)
                if any(area == Area.STASH for area, _ in locations):
                    status.stash = {card for card in status.stash if card.color != color}

                # c) 增加已使用的 Stash 计数 (只加一次!)
                status.used_stash += 1
                
                # 状态已改变，必须从头重新检查 (新暴露的牌可能是特殊牌)
                return check_flower_card(status)

    # 如果代码运行到这里，说明没有特殊牌，也没有凑齐的花牌
    # 状态是稳定的，函数结束。
    # (原函数没有返回值，这里保持一致)