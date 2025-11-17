
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
        if self.type() == CardType.NUMBER:
            return self.value // 9
        elif self.type() == CardType.FLOWER:
            return (self.value - 27) // 4
        else:
            raise ValueError

    @property
    def num(self) -> int:
        if self.type() == CardType.NUMBER:
            return self.value % 9 + 1
        else:
            return 0


class Stack:
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

    def count_continuous_length(self):
        if len(self.cards) < 2:
            return
        for idx in reversed(range(1, len(self.cards) - self.continuous_length)):
            if self.cards[idx].type == CardType.NUMBER and self.cards[idx-1].type == CardType.NUMBER \
                  and self.cards[idx].color != self.cards[idx-1].color \
                  and self.cards[idx].num == self.cards[idx-1].num - 1:
                self.continuous_length += 1
            else:
                break
    
    def move(self, count) -> List[Card]:
        ret: List[Card] = list()
        # 移除旧的卡片
        for _ in range(count):
            ret.append(self.cards.pop())
        self.continuous_length = max(0, self.continuous_length - count)
        if self.continuous_length == 0:
            self.count_continuous_length()
        return reversed(ret)
        
    def pop(self) -> Card:
        card = self.cards.pop()
        self.continuous_length = max(0, self.continuous_length - 1)
        if self.continuous_length == 0:
            self.count_continuous_length()
        return card

    def top(self) -> Optional[Card]:
        if len(self.cards):
            return self.cards[-1]
        else:
            return None
    
    def append(self, card: Card):
        self.cards.append(card)


class Status:
    def __init__(self):
        self.unused_card: List[int] = [1] * 40
        self.stacks: List[Stack] = [Stack() for _ in range(8)]
        self.used_stash = 0
        self.stash: Set[Card] = Set()
        self.after_sorting: List[Stack] = list(Stack() for _ in range(3))

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