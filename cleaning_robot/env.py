"""Cleaning robot vs. dog MDP.

Grid coordinates are (x, y) with x growing to the right (east) and y growing
downwards (south). Directions are indexed N=0, E=1, S=2, W=3.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import cached_property

import numpy as np

N, E, S, W, STAY = 0, 1, 2, 3, 4
ACTIONS = (N, E, S, W, STAY)
ACTION_NAMES = ("N", "E", "S", "W", "STAY")
DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))

Pos = tuple[int, int]


@dataclass(frozen=True)
class State:
    robot_pos: Pos
    dog_pos: Pos
    dog_face: int
    dog_playing: bool
    mess_pos: Pos | None


@dataclass(frozen=True)
class Room:
    width: int
    height: int
    obstacles: frozenset[Pos]

    @classmethod
    def symmetric(cls, width, height, obs_w, obs_h, wall_gap, stagger=0):
        """Two obs_w x obs_h obstacles, `wall_gap` cells away from the side walls and
        vertically centered, then the left one moved `stagger` cells up and the right
        one `stagger` cells down."""
        y0 = (height - obs_h) // 2
        corners = ((wall_gap, y0 - stagger), (width - wall_gap - obs_w, y0 + stagger))
        obstacles = frozenset(
            (x0 + dx, y + dy) for x0, y in corners for dx in range(obs_w) for dy in range(obs_h)
        )
        return cls(width, height, obstacles)

    @classmethod
    def centered(cls, width, height, obs_w, obs_h):
        """A single obs_w x obs_h obstacle (the dining table) in the middle of the room."""
        x0, y0 = (width - obs_w) // 2, (height - obs_h) // 2
        obstacles = frozenset((x0 + dx, y0 + dy) for dx in range(obs_w) for dy in range(obs_h))
        return cls(width, height, obstacles)

    def is_free(self, pos: Pos) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height and pos not in self.obstacles


SMALL_ROOM = Room.centered(9, 5, obs_w=3, obs_h=1)
README_ROOM = Room.symmetric(15, 8, obs_w=4, obs_h=2, wall_gap=2)


def _sample(outcomes, u: float):
    """Pick from (outcome, probability) pairs using a uniform number u in [0, 1)."""
    for outcome, pr in outcomes:
        u -= pr
        if u < 0:
            return outcome
    return outcome  # guard against rounding


class CleaningEnv:
    """Robot cleaning dog pee while avoiding being spotted.

    p: probability the wandering dog stops at each round (then turns to one of the
       other 3 directions and pees if the room is clean).
    k: number of rounds the dog wanders freely while the robot reboots after a capture.
    The robot reboots where it was caught.
    """

    def __init__(self, room: Room = SMALL_ROOM, p: float = 0.2, k: int = 2):
        self.room = room
        self.p = p
        self.k = k
        self.cells: list[Pos] = [
            (x, y) for y in range(room.height) for x in range(room.width) if room.is_free((x, y))
        ]
        self.index = {c: i for i, c in enumerate(self.cells)}
        self.n_cells = len(self.cells)
        self._wander_cache: dict[tuple[int, int, int, int], dict[tuple[int, int, int], float]] = {}

    # ------------------------------------------------------------------ geometry

    @cached_property
    def move(self) -> np.ndarray:
        """move[c, d] = cell reached moving from c in direction d, or -1 if blocked."""
        m = np.full((self.n_cells, 4), -1, dtype=np.int64)
        for i, (x, y) in enumerate(self.cells):
            for d, (dx, dy) in enumerate(DIRS):
                m[i, d] = self.index.get((x + dx, y + dy), -1)
        return m

    @cached_property
    def line_of_sight(self) -> np.ndarray:
        """los[a, b]: segment between cell centers touches no obstacle (corners block)."""
        F, eps = self.n_cells, 1e-9
        los = np.ones((F, F), dtype=bool)
        for a, (x0, y0) in enumerate(self.cells):
            for b in range(a + 1, F):
                x1, y1 = self.cells[b]
                n = 8 * max(abs(x1 - x0), abs(y1 - y0)) + 1
                for t in np.linspace(0.0, 1.0, n):
                    px, py = x0 + t * (x1 - x0) + 0.5, y0 + t * (y1 - y0) + 0.5
                    xs = {int(np.floor(px - eps)), int(np.floor(px + eps))}
                    ys = {int(np.floor(py - eps)), int(np.floor(py + eps))}
                    if any((cx, cy) in self.room.obstacles for cx in xs for cy in ys):
                        los[a, b] = los[b, a] = False
                        break
        return los

    @cached_property
    def sees(self) -> np.ndarray:
        """sees[dog, face, robot]: robot inside the dog's 90-degree view cone, unobstructed,
        or moving onto the dog's own cell."""
        F = self.n_cells
        out = np.zeros((F, 4, F), dtype=bool)
        for a, (x0, y0) in enumerate(self.cells):
            for f, (fx, fy) in enumerate(DIRS):
                for b, (x1, y1) in enumerate(self.cells):
                    dx, dy = x1 - x0, y1 - y0
                    forward = dx * fx + dy * fy
                    lateral = abs(dx * fy - dy * fx)
                    in_cone = forward > 0 and lateral <= forward and self.line_of_sight[a, b]
                    out[a, f, b] = in_cone or a == b
        return out

    def _king_neighbors(self, c: int) -> list[int]:
        """8-neighbors of c; diagonals only when both orthogonal cells are free."""
        x, y = self.cells[c]
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (dx, dy) == (0, 0) or (x + dx, y + dy) not in self.index:
                    continue
                if dx and dy and ((x + dx, y) not in self.index or (x, y + dy) not in self.index):
                    continue
                out.append(self.index[(x + dx, y + dy)])
        return out

    def _bfs(self, neighbors) -> np.ndarray:
        F = self.n_cells
        dist = np.full((F, F), np.iinfo(np.int64).max, dtype=np.int64)
        for src in range(F):
            dist[src, src] = 0
            queue = deque([src])
            while queue:
                c = queue.popleft()
                for n in neighbors(c):
                    if dist[src, n] > dist[src, c] + 1:
                        dist[src, n] = dist[src, c] + 1
                        queue.append(n)
        return dist

    @cached_property
    def king_dist(self) -> np.ndarray:
        return self._bfs(self._king_neighbors)

    @cached_property
    def grid_dist(self) -> np.ndarray:
        """4-connected shortest-path distance (robot moves)."""
        return self._bfs(lambda c: [n for n in self.move[c] if n >= 0])

    @cached_property
    def chase(self) -> np.ndarray:
        """chase[dog, target] = (new dog cell, new face) after two king steps towards target.
        Face is -1 when the dog lands on the target (keep current face)."""
        F = self.n_cells
        out = np.zeros((F, F, 2), dtype=np.int64)
        for dog in range(F):
            for tgt in range(F):
                c = dog
                for _ in range(2):
                    if c == tgt:
                        break
                    tx, ty = self.cells[tgt]

                    def key(n):
                        nx, ny = self.cells[n]
                        return (self.king_dist[n, tgt], (nx - tx) ** 2 + (ny - ty) ** 2)

                    c = min([c] + self._king_neighbors(c), key=key)
                out[dog, tgt] = (c, self._face_towards(c, tgt))
        return out

    def _face_towards(self, a: int, b: int) -> int:
        (x0, y0), (x1, y1) = self.cells[a], self.cells[b]
        dx, dy = x1 - x0, y1 - y0
        if dx == dy == 0:
            return -1
        if abs(dx) >= abs(dy):
            return E if dx > 0 else W
        return S if dy > 0 else N

    # ------------------------------------------------------------------ dynamics
    # Internal states are tuples (robot, dog, face, playing, mess) of ints, mess = -1 if clean.

    def _wander(self, dog: int, face: int, mess: int) -> list[tuple[tuple[int, int, int], float]]:
        """One round of the dog in normal mode."""
        out = []
        if self.p < 1:
            nxt = self.move[dog, face]
            out.append(((dog, (face + 2) % 4, mess) if nxt < 0 else (int(nxt), face, mess), 1 - self.p))
        if self.p > 0:
            new_mess = dog if mess < 0 else mess
            out += [((dog, f, new_mess), self.p / 3) for f in range(4) if f != face]
        return out

    def _wander_k(self, dog: int, face: int, mess: int, k: int) -> dict[tuple[int, int, int], float]:
        """Distribution of (dog, face, mess) after k rounds of normal mode."""
        key = (dog, face, mess, k)
        if key not in self._wander_cache:
            if k == 0:
                dist = {(dog, face, mess): 1.0}
            else:
                dist = {}
                for (d, f, m), pr in self._wander(dog, face, mess):
                    for nxt, pr2 in self._wander_k(d, f, m, k - 1).items():
                        dist[nxt] = dist.get(nxt, 0.0) + pr * pr2
            self._wander_cache[key] = dist
        return self._wander_cache[key]

    def _play(self, dog: int, face: int, robot: int) -> tuple[int, int, bool]:
        """One round of the dog in play mode: returns (dog, face, still_playing)."""
        if self.line_of_sight[dog, robot]:
            c, f = self.chase[dog, robot]
            return int(c), (face if f < 0 else int(f)), True
        # Robot hidden: keep running in the direction it was last seen.
        c = dog
        for _ in range(2):
            if self.move[c, face] < 0:
                break
            c = int(self.move[c, face])
        if self.move[c, face] < 0 and not self.line_of_sight[c, robot]:
            return c, face, False
        return c, face, True

    def _transitions(self, s: tuple, a: int) -> tuple[dict[tuple, float], str | None, tuple | None]:
        """Next-state distribution, the (deterministic) event of the round and, on a
        capture, the state at the start of the k reboot rounds."""
        robot, dog, face, playing, mess = s
        nxt = self.move[robot, a] if a != STAY else -1
        moved = nxt >= 0
        if moved:
            robot = int(nxt)
        if mess == robot:
            mess = -1

        event = None
        if not playing and moved and self.sees[dog, face, robot]:
            playing, event = True, "spotted"
        if playing:
            dog, face, playing = self._play(dog, face, robot)
            if dog == robot:
                dist = self._wander_k(dog, face, mess, self.k)
                start = (robot, dog, face, False, mess)
                return {(robot, d, f, False, m): pr for (d, f, m), pr in dist.items()}, "captured", start
            if not playing:
                event = "lost"
            return {(robot, dog, face, playing, mess): 1.0}, event, None

        out: dict[tuple, float] = {}
        for (d, f, m), pr in self._wander(dog, face, mess):
            key = (robot, d, f, False, m)
            out[key] = out.get(key, 0.0) + pr
        return out, event, None

    # ------------------------------------------------------------------ conversions

    def encode(self, state: State) -> tuple:
        mess = -1 if state.mess_pos is None else self.index[state.mess_pos]
        return (self.index[state.robot_pos], self.index[state.dog_pos], state.dog_face, state.dog_playing, mess)

    def decode(self, s: tuple) -> State:
        robot, dog, face, playing, mess = s
        return State(self.cells[robot], self.cells[dog], face, bool(playing), None if mess < 0 else self.cells[mess])

    # ------------------------------------------------------------------ MDP interface

    def reward(self, state: State) -> float:
        return -1.0 if state.mess_pos is not None else 0.0

    def next_state_probs(self, state: State, action: int) -> dict[State, float]:
        dist, _, _ = self._transitions(self.encode(state), action)
        return {self.decode(s): pr for s, pr in dist.items()}

    def step(self, state: State, action: int, seed=None) -> tuple[State, str | None, list[State]]:
        """Sample the next state. Also returns the round's event (None, "spotted",
        "lost" or "captured") and, on a capture, the reboot trajectory: the state when
        the reboot starts followed by the k states the dog wanders through (the last one
        is the next state). Sampling the k rounds one by one has the same distribution
        as next_state_probs."""
        rng = np.random.default_rng(seed)
        dist, event, start = self._transitions(self.encode(state), action)
        if start is None:
            return self.decode(_sample(dist.items(), rng.random())), event, []
        robot, *_ = start
        path = [start]
        for _ in range(self.k):
            _, d, f, _, m = path[-1]
            d, f, m = _sample(self._wander(d, f, m), rng.random())
            path.append((robot, d, f, False, m))
        return self.decode(path[-1]), event, [self.decode(x) for x in path]

    def next_state_sample(self, state: State, action: int, seed=None) -> State:
        return self.step(state, action, seed)[0]

    def random_init(self, seed=None) -> State:
        """Uniform robot, dog, dog face and mess (possibly clean); dog not playing."""
        rng = np.random.default_rng(seed)
        F = self.n_cells
        robot, dog = rng.integers(F, size=2)
        mess = int(rng.integers(-1, F))
        return self.decode((int(robot), int(dog), int(rng.integers(4)), False, mess))
