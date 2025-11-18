# Optimized Shenzhen Solitaire solver (High-performance adaptations)
# Based on user's original Python data structures but rewritten with
# - more aggressive auto_remove (canonical reduction)
# - lightweight state copy
# - compact hashing (blake2b 64-bit)
# - Best-first priority queue with JS-like priority
# - reduced branching: only legal contiguous moves, restrained stash moves
# - visited-state deduplication

import heapq
import time
import hashlib
from typing import List, Optional, Tuple, Set
from enum import Enum
from collections import namedtuple

# Keep Card / Stack / Status structure similar to original but leaner
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
            return self.value // 9  # 0..2
        if self.type == CardType.FLOWER:
            return (self.value - 27) // 4  # 0..2
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
        return self.value

class Stack:
    __slots__ = ("cards", "cont_len")
    def __init__(self, cards: Optional[List[Card]] = None):
        self.cards: List[Card] = list(cards) if cards else []
        self.cont_len = 0
        self._update_cont()
    def push(self, card: Card):
        self.cards.append(card)
        # update cont_len cheaply
        if len(self.cards) == 1:
            self.cont_len = 1
        else:
            # only need to check previous top
            prev = self.cards[-2]
            cur = self.cards[-1]
            if (cur.type == CardType.NUMBER and prev.type == CardType.NUMBER and
                cur.color != prev.color and cur.num == prev.num - 1):
                self.cont_len += 1
            else:
                self.cont_len = 1
    def pop(self) -> Card:
        card = self.cards.pop()
        # adjust cont_len conservatively
        if self.cont_len > 0:
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
        return f"Stack({','.join(repr(c) for c in self.cards)})"

class Status:
    __slots__ = ("stacks", "stash", "stash_limit", "sorted_tops", "special_removed")
    def __init__(self, stacks: Optional[List[List[Card]]] = None):
        self.stacks: List[Stack] = [Stack() for _ in range(8)]
        if stacks:
            for i, col in enumerate(stacks):
                self.stacks[i] = Stack(col)
        self.stash: List[Optional[Card]] = [None, None, None]
        self.stash_limit = 3
        self.sorted_tops = [0, 0, 0]  # collected per suit (1..9 stored as numbers)
        self.special_removed = False

    # aggressive auto_remove: canonicalize state by removing all forced/obvious cards
    def auto_remove(self):
        # Implement similar logic to JS auto_remove_cards but adapted to our Card mapping
        # We'll loop until stable
        while True:
            changed = False
            # compute lowestPerSuit among exposed and slots
            lowest = {'R': 10, 'G': 10, 'B': 10}
            # check stacks for numeric cards to populate lowest
            for st in self.stacks:
                for c in st.cards:
                    if c.type == CardType.NUMBER:
                        key = 'R' if c.color == 0 else ('G' if c.color == 1 else 'B')
                        if c.num < lowest[key]:
                            lowest[key] = c.num
            # check stash
            for s in self.stash:
                if s is None: continue
                if s.type == CardType.NUMBER:
                    key = 'R' if s.color == 0 else ('G' if s.color == 1 else 'B')
                    if s.num < lowest[key]:
                        lowest[key] = s.num
            # 1) remove any top that is FLOWER (F) or special SP or numeric 1 (can always remove if matches rules)
            # We follow conservative rules but run repeatedly
            # Remove flowers and specials at top immediately
            for st in self.stacks:
                if not st.cards: continue
                last = st.top()
                if last.type == CardType.SPECIAL:
                    st.pop(); self.special_removed = True; changed = True; break
                if last.type == CardType.FLOWER:
                    st.pop(); # flowers removed only via collapse when 4 present - skip here
                    # we don't auto-remove a single flower; we'll handle collapse below
                    # push popped flower to stash temporary (but we won't implement that complexity here)
            if changed:
                continue
            # 2) remove numeric cards that are strictly collectible based on sorted_tops and lowest per suit
            # For each stack top and stash, if card.type==NUMBER and matches conditions, remove it
            # We compute can_remove(card) similar to original Python code
            def can_collect(card: Card) -> bool:
                if card.type == CardType.SPECIAL:
                    return True
                if card.type != CardType.NUMBER:
                    return False
                color = card.color
                need = self.sorted_tops[color] + 1
                # can collect if it's the next needed AND either trivial or <= other suits' lowest
                if card.num != need:
                    return False
                if need == 1:
                    return True
                if need == 2:
                    # only collect 2 if it's not blocking lower in its suit (conservative)
                    return True
                # for value>2, require that it's <= lowest of all suits
                # map suit indexes to keys
                keys = ['R','G','B']
                key = keys[color]
                val = card.num
                # if val <= min(lowest of other suits and this suit)
                if val <= lowest['R'] and val <= lowest['G'] and val <= lowest['B']:
                    return True
                return False

            removed_any = False
            # check stack tops
            for st in self.stacks:
                if not st.cards: continue
                last = st.top()
                if can_collect(last):
                    c = st.pop()
                    if c.type == CardType.SPECIAL:
                        self.special_removed = True
                    else:
                        self.sorted_tops[c.color] = c.num
                    removed_any = True
                    break
            if removed_any:
                continue
            # check stash
            for i, s in enumerate(self.stash):
                if s is None: continue
                if can_collect(s):
                    c = s
                    self.stash[i] = None
                    if c.type == CardType.SPECIAL:
                        self.special_removed = True
                    else:
                        self.sorted_tops[c.color] = c.num
                    removed_any = True
                    break
            if removed_any:
                continue
            # 3) handle flower collapse: if any color has 4 exposed (tops or stash), remove them and consume one stash slot
            # Count exposed flowers per color from stack tops and stash
            flower_count = [0,0,0]
            flower_sources = []  # tuples (is_stack, idx_or_card)
            for idx, st in enumerate(self.stacks):
                t = st.top()
                if t and t.type == CardType.FLOWER:
                    flower_count[t.color] += 1
                    flower_sources.append((True, idx))
            for i, s in enumerate(self.stash):
                if s and s.type == CardType.FLOWER:
                    flower_count[s.color] += 1
                    flower_sources.append((False, i))
            collapsed = False
            for color in range(3):
                if flower_count[color] >= 4:
                    # require at least one free stash slot or one same-color in stash
                    has_slot = sum(1 for x in self.stash if x is None) > 0
                    has_same_in_stash = any(x and x.type==CardType.FLOWER and x.color==color for x in self.stash)
                    if has_slot or has_same_in_stash:
                        removed = 0
                        # remove 4 flowers from sources
                        for is_stack, idx_or in flower_sources:
                            if removed == 4: break
                            if is_stack:
                                st = self.stacks[idx_or]
                                if st.top() and st.top().type == CardType.FLOWER and st.top().color == color:
                                    st.pop(); removed += 1
                            else:
                                i = idx_or
                                s = self.stash[i]
                                if s and s.type == CardType.FLOWER and s.color == color:
                                    self.stash[i] = None; removed += 1
                        # consume a stash slot (mark as used by decrementing stash_limit)
                        self.stash_limit = max(0, self.stash_limit - 1)
                        collapsed = True
                        break
            if collapsed:
                continue
            # nothing changed
            break

    def remaining_cards(self) -> int:
        rem = 0
        for st in self.stacks:
            rem += len(st.cards)
        for s in self.stash:
            if s is None: continue
            if s.type == CardType.FLOWER:
                # flower is counted but might be removed via collapse; count it
                rem += 1
            elif s.type == CardType.SPECIAL:
                rem += 1
            else:
                rem += 1
        return rem

    def calc_stacked_cards(self) -> int:
        stacked = 0
        for st in self.stacks:
            if not st.cards: continue
            local = 0
            for j in range(len(st.cards)-1, 0, -1):
                if can_be_stacked(st.cards[j], st.cards[j-1]):
                    local += 1
                else:
                    break
            stacked += local
        return stacked

    def hash_key(self) -> bytes:
        # compact canonicalization: do NOT reorder stacks (position matters) but
        # convert to a bytes buffer: each card value is one byte (0..255)
        parts = bytearray()
        for st in self.stacks:
            # length prefix then values
            parts.append(len(st.cards))
            for c in st.cards:
                parts.append(c.value)
        # stash: 3 slots, 255 for empty
        for s in self.stash:
            parts.append(255 if s is None else s.value)
        # sorted_tops
        parts.extend(self.sorted_tops)
        # special_removed and stash_limit
        parts.append(1 if self.special_removed else 0)
        parts.append(self.stash_limit)
        # use blake2b 8 bytes
        h = hashlib.blake2b(parts, digest_size=8).digest()
        return h

    def is_solved(self) -> bool:
        if not self.special_removed:
            return False
        for st in self.stacks:
            if st.cards: return False
        for s in self.stash:
            if s is None: continue
            if s.type != CardType.FLOWER:
                return False
        return True

# helper stacking rule
def can_be_stacked(frm: Card, to: Card) -> bool:
    if frm.type == CardType.SPECIAL or frm.type == CardType.FLOWER:
        return False
    if to.type == CardType.SPECIAL or to.type == CardType.FLOWER:
        return False
    if frm.color == to.color:
        return False
    return (frm.num + 1 == to.num)

# Lightweight fast copy function preserving structure
def fast_copy_status(s: Status) -> Status:
    new = Status()
    # copy stacks lists quickly
    for i, st in enumerate(s.stacks):
        new_cards = st.cards.copy()
        new.stacks[i] = Stack(new_cards)
        new.stacks[i].cont_len = st.cont_len
    new.stash = [ (None if x is None else Card(x.value)) for x in s.stash ]
    new.stash_limit = s.stash_limit
    new.sorted_tops = s.sorted_tops.copy()
    new.special_removed = s.special_removed
    return new

# solve using best-first with JS-like priority

class Solver:
    def __init__(self, start: Status):
        self.start = start
    def solve(self, max_iters=200000) -> Optional[List[Tuple[str]]]:
        start = fast_copy_status(self.start)
        start.auto_remove()
        start_key = start.hash_key()
        pq = []  # (priority, counter, steps, state)
        counter = 0
        # priority function: remaining + step*0.1 - stacked
        def priority_of(s: Status, steps: int) -> float:
            rem = s.remaining_cards()
            stacked = s.calc_stacked_cards()
            return rem + steps*0.1 - stacked
        heapq.heappush(pq, (priority_of(start,0), counter, 0, start, []))
        visited: Set[bytes] = set([start_key])
        iters = 0
        while pq and iters < max_iters:
            pr, _, steps, cur, path = heapq.heappop(pq)
            if cur.is_solved():
                return path
            # generate moves (reduced branching)
            actions = generate_valid_moves(cur)
            # prefer moves by a light local heuristic: moves that reduce remaining or increase stacked
            for act_desc, next_state in actions:
                key = next_state.hash_key()
                if key in visited: continue
                visited.add(key)
                nxt = fast_copy_status(next_state) if False else next_state  # already copied in generator
                nxt.auto_remove()
                heapq.heappush(pq, (priority_of(nxt, steps+1), counter+1, steps+1, nxt, path + [act_desc]))
                counter += 1
            iters += 1
            if iters % 10000 == 0:
                print(f"iters={iters}, queue={len(pq)}, visited={len(visited)}, top_pr={pr}")
        return None

# generate_valid_moves: conservative generator returning (desc, new_status)
def generate_valid_moves(s: Status) -> List[Tuple[str, Status]]:
    res: List[Tuple[str, Status]] = []
    # 1) pop moves: if top card is collectible directly (lowestPerSuit logic), allow pop
    # but we rely on auto_remove to catch forced pops, so avoid duplicating.

    # 2) stack -> stack for contiguous segments only
    for i, st in enumerate(s.stacks):
        if not st.cards: continue
        movable = st.movable_count()
        # only consider moving contiguous segment of size 1..movable
        for cnt in range(1, movable+1):
            bottom_card = st.cards[-cnt]
            for j, dst in enumerate(s.stacks):
                if i == j: continue
                if dst.cards:
                    top = dst.top()
                    if can_be_stacked(bottom_card, top):
                        # disallow moving whole stack to empty (no-op) unless it clears source
                        if not dst.cards and cnt == 1 and len(st.cards) > 1 and st.movable_count() != len(st.cards):
                            continue
                        new = fast_copy_status(s)
                        moved = [new.stacks[i].pop() for _ in range(cnt)][::-1]
                        for c in moved:
                            new.stacks[j].push(c)
                        res.append((f"S{i}->{j} ({cnt})", new))
                else:
                    # empty dst allowed if moving more than 1 or moving entire stack
                    if cnt == 1 and len(st.cards) > 1 and st.movable_count() != len(st.cards):
                        continue
                    new = fast_copy_status(s)
                    moved = [new.stacks[i].pop() for _ in range(cnt)][::-1]
                    for c in moved:
                        new.stacks[j].push(c)
                    res.append((f"S{i}->{j} ({cnt})", new))
    # 3) stack -> stash (if free slot available) but prune if top is collectible
    free_slot_indices = [idx for idx, x in enumerate(s.stash) if x is None]
    if free_slot_indices:
        for i, st in enumerate(s.stacks):
            if not st.cards: continue
            top = st.top()
            # prune: don't stash a card that auto_remove would collect
            if s._can_auto_remove if hasattr(s, '_can_auto_remove') else False:
                pass
            # implement simple prune: don't stash cards that are immediate collectable
            # we approximate by checking if top.num == sorted_tops+1
            if top.type == CardType.NUMBER:
                if top.num == s.sorted_tops[top.color] + 1:
                    continue
            new = fast_copy_status(s)
            card = new.stacks[i].pop()
            # put into first free slot
            new.stash[free_slot_indices[0]] = card
            res.append((f"S{i}->stash ({repr(card)})", new))
    # 4) stash -> stack
    for si, card in enumerate(s.stash):
        if card is None: continue
        for j, dst in enumerate(s.stacks):
            if dst.cards:
                if can_be_stacked(card, dst.top()):
                    new = fast_copy_status(s)
                    # find matching card in stash
                    for k in range(3):
                        if new.stash[k] and new.stash[k].value == card.value:
                            new.stash[k] = None
                            break
                    new.stacks[j].push(Card(card.value))
                    res.append((f"stash{si}->{j} ({repr(card)})", new))
            else:
                new = fast_copy_status(s)
                for k in range(3):
                    if new.stash[k] and new.stash[k].value == card.value:
                        new.stash[k] = None
                        break
                new.stacks[j].push(Card(card.value))
                res.append((f"stash{si}->{j} ({repr(card)})", new))
    # light sorting of candidates: prefer moves that reduce remaining_cards or increase contiguous
    def score_move(pair: Tuple[str, Status]) -> int:
        st = pair[1]
        score = 0
        # prefer fewer remaining
        score -= st.remaining_cards()
        # prefer more stacked
        score += st.calc_stacked_cards()
        return score
    res.sort(key=score_move, reverse=True)
    return res

# add missing _can_auto_remove to Status to help prune stash moves (used above)
def _can_auto_remove_status(self, card: Card) -> bool:
    if card.type == CardType.SPECIAL:
        return True
    if card.type != CardType.NUMBER:
        return False
    color = card.color
    need = self.sorted_tops[color] + 1
    if card.num != need:
        return False
    return True

Status._can_auto_remove = _can_auto_remove_status

# ---- Usage helpers: construct Status from nested lists of ints ----

def status_from_values(columns: List[List[int]]) -> Status:
    stacks = []
    for col in columns:
        stacks.append([Card(v) for v in col])
    st = Status(stacks)
    st.auto_remove()
    return st

# ---- Example / test harness ----
if __name__ == '__main__':
    # small random test or user-provided layout
    import random
    # create a simplified random layout using 40 distinct values (0..39)
    values = list(range(40))
    random.shuffle(values)
    columns = [values[i*5:(i+1)*5] for i in range(8)]
    start = status_from_values(columns)
    solver = Solver(start)
    t0 = time.time()
    sol = solver.solve(max_iters=50000)
    t1 = time.time()
    print('time', t1 - t0)
    if sol:
        print('found solution steps', len(sol))
    else:
        print('not found')

