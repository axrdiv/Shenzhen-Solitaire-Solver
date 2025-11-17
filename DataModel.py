
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
        
    def pop(self, count = 1):
        ret: List[Card] = list()

        # 移除旧的卡片
        for _ in range(count):
            ret.append(self.cards.pop())
        
        self.continuous_length = max(0, self.continuous_length - count)
        self.count_continuous_length()

        return Stack(cards_list=reversed(ret))

    def top(self) -> Optional[Card]:
        if len(self.cards):
            return self.cards[-1]
        else:
            return None


class Status:
    def __init__(self):
        self.unused_card: List[int] = [1] * 40
        self.stacks: List[Stack] = [Stack() for _ in range(8)]
        self.used_stash = 0
        self.stash: Set[Card] = Set()
        self.after_sorting: List[Stack] = list(Stack() for _ in range(3))


def check_flower_card(status: Status):
    flower_card_count_exposed: List[Set] = [set() for _ in range(3)]
    for idx, st in enumerate(status.stacks):
        if len(st.cards):
            top_card = st.top()
            # 移除特殊牌
            if top_card.type == CardType.SPECIAL:
                st.pop()
                return check_flower_card(status)

            if top_card.type == CardType.FLOWER:
                flower_card_count_exposed[top_card.color].add((Area.STACK, idx))
    
    # 检查 stash 中是否有花色牌
    for card in status.stash:
        if card.type == CardType.FLOWER:
            flower_card_count_exposed[card.color].add((Area.STASH, card.value))

    
    for color, flower_card_count in enumerate(flower_card_count_exposed):
        # 移除花色牌
        if len(flower_card_count) == 4:
            if len(status.stash) < 3 - status.used_stash:
                for area, val in flower_card_count:
                    if area == Area.STACK:
                        status.stacks[val].pop()
                    else:
                        new_set = set() 
                        for card in status.stash:
                            # 移除这个花色
                            if card.color == color:
                                continue
                            new_set.add(card)
                        del status.stash
                        status.stash = new_set
                        status.used_stash += 1
            else:
                for card in status.stash:
                    # 该花色在 stash 区有位置
                    if card.color == color:
                        for area, val in flower_card_count:
                            if area == Area.STACK:
                                status.stacks[val].pop()
                            else:
                                new_set = set() 
                                for card in status.stash:
                                    # 移除这个花色
                                    if card.color == color:
                                        continue
                                    new_set.add(card)
                                del status.stash
                                status.stash = new_set
                                status.used_stash += 1
        


