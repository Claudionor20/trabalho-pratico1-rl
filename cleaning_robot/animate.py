"""Pixel-art (old-school handheld RPG style) rendering of the cleaning robot MDP."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .env import DIRS, E, N, S, W, CleaningEnv, State
from .policy import PolicyMixin

T = 16  # tile size in pixels

PALETTE = {
    "k": (26, 28, 44),  # outline
    "w": (244, 244, 244),
    "l": (196, 204, 214),  # light gray
    "g": (130, 140, 156),  # gray
    "d": (86, 94, 112),  # dark gray
    "b": (65, 166, 246),  # led blue
    "c": (115, 239, 247),  # cyan
    "r": (177, 62, 83),  # red
    "p": (239, 125, 160),  # pink
    "o": (205, 130, 70),  # dog orange-brown
    "O": (150, 85, 45),  # dog dark brown
    "t": (240, 205, 150),  # dog tan
    "y": (255, 205, 117),  # pee
    "Y": (230, 170, 60),  # pee dark
    "h": (255, 250, 200),  # pee highlight
}

ROBOT = [
    "................",
    ".....kkkkkk.....",
    "...kkllllllkk...",
    "..klllwwwwlllk..",
    "..kllwwwwwwllk..",
    ".klwwwwwwwwwwlk.",
    ".klwwwkkkkwwwlk.",
    ".klwwkbccbkwwlk.",
    ".klwwkbbbbkwwlk.",
    ".klwwwkkkkwwwlk.",
    ".kglwwwwwwwwlgk.",
    "..kgllwwwwllgk..",
    "..kdggllllggdk..",
    "...kkddddddkk...",
    ".....kkkkkk.....",
    "................",
]

DOG_RIGHT = [
    "................",
    "................",
    "..........kk....",
    ".........kOOk...",
    "........kooOokk.",
    "........kokooook",
    "..kk....koooootk",
    ".kOk....kotttkk.",
    "..kok...kooook..",
    "...kokkkkooook..",
    "...kooooooooOk..",
    "...kooootttook..",
    "...koOkkkkkoOk..",
    "...kok....kok...",
    "...kk.....kk....",
    "................",
]

DOG_DOWN = [
    "................",
    "...kk......kk...",
    "..kOOk....kOOk..",
    "..kOokkkkkkoOk..",
    "...koooooooook..",
    "...kokooookok...",
    "...koootttook...",
    "....kotkktok....",
    ".....kttttk.....",
    "....kookkook....",
    "...kooooooook...",
    "...kootttook....",
    "...kooooooook...",
    "...kokkkkkkok...",
    "...kk......kk...",
    "................",
]

DOG_UP = [
    "................",
    "...kk......kk...",
    "..kOOk....kOOk..",
    "..kOokkkkkkoOk..",
    "...koooooooook..",
    "...koooooooook..",
    "...koooooooook..",
    "....koooooook...",
    ".....kkooook....",
    "....kookkook....",
    "...koooOOoook...",
    "...kooOkkOook...",
    "...koooOOoook...",
    "...kokkkkkkok...",
    "...kk......kk...",
    "................",
]

PEE = [
    "................",
    "................",
    "................",
    "................",
    "................",
    "......YYYY......",
    "....YYyyyyYY....",
    "...YyyhhyyyyY...",
    "..YyyhyyyyyyyY..",
    "..YyyyyyyyyyyY..",
    "...YyyyyyyyhyY..",
    "....YYyyyyyYY...",
    "......YYYYY.....",
    "................",
    "................",
    "................",
]

BUBBLE = [
    "..kkkkkkkk..",
    ".kwwwwwwwwk.",
    ".kwwwrrwwwk.",
    ".kwwwrrwwwk.",
    ".kwwwrrwwwk.",
    ".kwwwrrwwwk.",
    ".kwwwwwwwwk.",
    ".kwwwrrwwwk.",
    ".kwwwwwwwwk.",
    "..kkkwkkkk..",
    "....kwk.....",
    ".....k......",
]

HEART = [
    ".kk.kk.",
    "kppkrrk",
    "kpprrrk",
    "krrrrrk",
    ".krrrk.",
    "..krk..",
    "...k...",
]


def _sprite(rows: list[str]) -> np.ndarray:
    """RGBA uint8 array from an ascii sprite ('.' is transparent)."""
    out = np.zeros((len(rows), len(rows[0]), 4), dtype=np.uint8)
    for i, row in enumerate(rows):
        for j, ch in enumerate(row):
            if ch != ".":
                out[i, j, :3] = PALETTE[ch]
                out[i, j, 3] = 255
    return out


SPRITES = {
    "robot": _sprite(ROBOT),
    "pee": _sprite(PEE),
    "bubble": _sprite(BUBBLE),
    "heart": _sprite(HEART),
    "dog": {
        E: _sprite(DOG_RIGHT),
        W: _sprite(DOG_RIGHT)[:, ::-1],
        S: _sprite(DOG_DOWN),
        N: _sprite(DOG_UP),
    },
}


def _floor_tile(x: int, y: int) -> np.ndarray:
    """Wooden planks, offset every other row."""
    tile = np.empty((T, T, 3), dtype=np.uint8)
    tile[:] = (193, 140, 96) if (x + y) % 2 else (186, 133, 90)
    tile[::4, :] = (150, 100, 66)
    off = 0 if y % 2 else 8
    for r in range(0, T, 4):
        seam = (off + 4 * (r // 4) * 3) % T
        tile[r : r + 4, seam] = (150, 100, 66)
    tile[1::4, :] = np.minimum(tile[1::4, :].astype(int) + 14, 255)
    return tile


OUTLINE = (26, 28, 44)

# Margins (pixels) around the room: garden + wall on each side.
WALL = 5  # thickness of the wall tops seen from above
FACE = T  # height of the inner face of the top wall
SIDING = 7  # outer face of the bottom wall
GARDEN = 18


def _garden(height: int, width: int, seed: int = 0) -> np.ndarray:
    """Grass with tufts and little flowers."""
    rng = np.random.default_rng(seed)
    g = np.empty((height, width, 3), dtype=np.uint8)
    g[:] = (106, 190, 48)
    g[rng.random((height, width)) < 0.12] = (88, 168, 40)
    for _ in range(height * width // 40):  # tufts: small "v"
        y, x = rng.integers(1, height - 1), rng.integers(1, width - 1)
        g[y, x - 1] = g[y, x + 1] = g[y + 1, x] = (60, 135, 40)
    colors = [(240, 90, 100), (255, 220, 90), (250, 250, 250), (180, 130, 230)]
    for _ in range(height * width // 160):  # flowers: 4 petals around a yellow center
        y, x = rng.integers(1, height - 1), rng.integers(1, width - 1)
        c = colors[rng.integers(len(colors))]
        g[y - 1, x] = g[y + 1, x] = g[y, x - 1] = g[y, x + 1] = c
        g[y, x] = (250, 200, 60)
    return g


def _window(canvas: np.ndarray, x: int, y: int) -> None:
    """Window on the inner face of the top wall (12 x 11 px), glass with reflections."""
    canvas[y : y + 11, x : x + 12] = OUTLINE
    canvas[y + 1 : y + 10, x + 1 : x + 11] = (240, 236, 220)  # frame
    glass = canvas[y + 2 : y + 9, x + 2 : x + 10]
    rows = np.linspace(0, 1, glass.shape[0])[:, None, None]
    glass[:] = ((1 - rows) * np.array((170, 205, 230)) + rows * np.array((100, 135, 175))).astype(np.uint8)
    for i in range(glass.shape[0]):  # two diagonal glints
        for j in (i - 1, i + 3):
            if 0 <= glass.shape[1] - 1 - j < glass.shape[1] and 0 <= j:
                glass[i, glass.shape[1] - 1 - j] = (235, 245, 255)
    canvas[y + 2 : y + 9, x + 5 : x + 7] = (240, 236, 220)  # mullion
    canvas[y + 5, x + 2 : x + 10] = (240, 236, 220)
    canvas[y + 10, x - 1 : x + 13] = (200, 190, 170)  # sill


def _door_and_walkway(canvas: np.ndarray, x: int, y_face: int, y_top: int) -> None:
    """Door (14 px wide) on the inner face of the top wall at column x, with a
    cobblestone walkway from the threshold to the top of the garden."""
    # walkway
    path = canvas[0:y_top, x - 1 : x + 15]
    path[:] = (196, 186, 166)
    for r in range(path.shape[0]):
        if r % 4 == 3:
            path[r, :] = (150, 140, 124)
        else:
            path[r, (2 if (r // 4) % 2 else 6) :: 8] = (150, 140, 124)
    path[:, [0, -1]] = (120, 112, 100)
    # door
    canvas[y_face : y_face + FACE, x : x + 14] = OUTLINE
    canvas[y_face + 1 : y_face + FACE, x + 1 : x + 13] = (178, 70, 62)
    canvas[y_face + 2 : y_face + 7, x + 3 : x + 11] = (150, 54, 50)  # panels
    canvas[y_face + 9 : y_face + 14, x + 3 : x + 11] = (150, 54, 50)
    canvas[y_face + 2 : y_face + 7, x + 3] = canvas[y_face + 9 : y_face + 14, x + 3] = (205, 100, 88)
    canvas[y_face + 8 : y_face + 10, x + 10 : x + 12] = (250, 210, 80)  # knob
    canvas[y_face + FACE - 1, x - 1 : x + 15] = (200, 190, 170)  # threshold


def _wall_face(face: np.ndarray, paper, stripe) -> None:
    """Inner face of a north wall: striped wallpaper, wainscot and baseboard."""
    face[:] = paper
    face[:, ::4] = stripe
    face[-7:] = (150, 96, 64)
    face[-7] = (196, 140, 96)
    face[-3:] = (110, 66, 44)


def _tiles(height: int, width: int, a, b, size: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    return np.where((((yy // size) + (xx // size)) % 2 == 0)[..., None], a, b).astype(np.uint8)


def _house(room) -> tuple[np.ndarray, int, int]:
    """Background: the room inside a bigger house that continues to the west and south,
    with the garden to the north and east. Returns the image and the pixel origin of
    room cell (0, 0)."""
    w, h = room.width * T, room.height * T
    ox, oy = GARDEN + WALL, GARDEN + WALL + FACE
    height, width = oy + h + WALL + SIDING + GARDEN, ox + w + WALL + GARDEN
    canvas = _garden(height, width)
    x_east = ox + w + WALL  # outer east edge of the house
    y_north = oy - FACE - WALL  # outer north edge of the house
    y_south = oy + h  # interior south wall (top)
    # shadow cast on the grass east of the house
    canvas[y_north + 3 :, x_east : x_east + 3] = (canvas[y_north + 3 :, x_east : x_east + 3] * 0.7).astype(np.uint8)
    # house footprint: wall tops (light stone) everywhere, then the rooms are carved out
    canvas[y_north:, :x_east] = OUTLINE
    canvas[y_north + 1 :, : x_east - 1] = (222, 210, 182)
    canvas[y_north + 1, : x_east - 1] = (240, 232, 210)

    def carve(y0, y1, x0, x1):
        """Outlined rectangle for a room area; returns the inner view."""
        canvas[max(y0 - 1, 0) : y1 + 1, max(x0 - 1, 0) : x1 + 1] = OUTLINE
        return canvas[y0:y1, x0:x1]

    # this room (its floor tiles are drawn by the Renderer)
    _wall_face(carve(oy - FACE, oy, ox, ox + w), (236, 214, 190), (222, 196, 170))
    carve(oy, y_south, ox, ox + w)
    # west neighbor: sage wallpaper, blue tiled floor, closed door to this room
    xw = ox - WALL - 1
    _wall_face(carve(oy - FACE, oy, 0, xw), (196, 214, 180), (176, 198, 160))
    carve(oy, y_south, 0, xw)[:] = _tiles(y_south - oy, xw, (170, 196, 214), (150, 178, 200), 8)
    yd = oy + (room.height // 2) * T + 1
    canvas[yd : yd + 14, xw:ox] = OUTLINE
    canvas[yd + 1 : yd + 13, xw + 1 : ox - 1] = (178, 70, 62)
    canvas[yd + 1 : yd + 13, xw + 2] = (205, 100, 88)
    canvas[yd + 6 : yd + 8, ox - 3 : ox - 1] = (250, 210, 80)  # knob
    # south neighbor (one room under both): north wall face, kitchen checkerboard floor
    ys = y_south + WALL
    _wall_face(carve(ys, ys + 11, 0, ox + w), (250, 236, 200), (238, 220, 180))
    carve(ys + 11, height, 0, ox + w)[:] = _tiles(height - ys - 11, ox + w, (240, 236, 226), (90, 120, 140), 6)

    door = room.width // 2
    for i in range(1, room.width, 3):
        if i != door:
            _window(canvas, ox + i * T + 2, oy - FACE - 2)
    _door_and_walkway(canvas, ox + door * T + 1, oy - FACE, y_north)
    return canvas, ox, oy


def _table_tile(floor: np.ndarray, left: bool, right: bool, top: bool, bottom: bool, centerpiece: bool) -> np.ndarray:
    """Segment of the dining table drawn over the floor; edges only on the table's outer sides."""
    outline, wood, wood_dark = (26, 28, 44), (176, 110, 62), (128, 74, 40)
    tile = floor.copy()
    y0, y1 = (1 if top else 0), (13 if bottom else 16)
    x0, x1 = (1 if left else 0), (15 if right else 16)
    if bottom:  # legs and shadow below the table top
        tile[13:16, x0:x1] = (tile[13:16, x0:x1] * 0.7).astype(np.uint8)
        for lx in ([x0 + 1] if left else []) + ([x1 - 3] if right else []):
            tile[13:16, lx : lx + 2] = wood_dark
            tile[15, lx : lx + 2] = outline
    tile[y0:y1, x0:x1] = outline
    tile[y0 + top : y1 - bottom, x0 + left : x1 - right] = wood
    tile[y1 - bottom - 2 : y1 - bottom, x0 + left : x1 - right] = wood_dark
    # gingham table cloth
    rows, cols = np.mgrid[3:11, x0 + left : x1 - right]
    check = ((rows // 2 + cols // 2) % 2).astype(bool)
    tile[3:11, x0 + left : x1 - right] = np.where(check[..., None], (210, 70, 80), (240, 236, 228))
    yy, xx = np.mgrid[0:T, 0:T]
    r2 = (yy - 6.5) ** 2 + (xx - 7.5) ** 2
    if centerpiece:  # vase with flowers
        tile[5:10, 6:10] = outline
        tile[6:9, 7:9] = (70, 120, 200)
        tile[(r2 <= 3.5) & (yy < 6)] = (90, 170, 90)
        for fy, fx, col in ((2, 5, (240, 120, 160)), (1, 8, (250, 220, 90)), (3, 10, (240, 120, 160))):
            tile[fy : fy + 2, fx : fx + 2] = col
    else:  # plate
        tile[r2 <= 12] = (150, 150, 160)
        tile[r2 <= 7] = (250, 250, 250)
        tile[r2 <= 2] = (225, 225, 230)
    return tile


def _blit(canvas: np.ndarray, sprite: np.ndarray, px: int, py: int) -> None:
    h, w = sprite.shape[:2]
    H, Wd = canvas.shape[:2]
    x0, y0, x1, y1 = max(px, 0), max(py, 0), min(px + w, Wd), min(py + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    sp = sprite[y0 - py : y1 - py, x0 - px : x1 - px]
    alpha = sp[..., 3:4] / 255.0
    region = canvas[y0:y1, x0:x1]
    region[:] = (alpha * sp[..., :3] + (1 - alpha) * region).astype(np.uint8)


class Renderer:
    def __init__(self, env: CleaningEnv, hud_height: int = 12):
        self.env = env
        room = env.room
        self.hud = hud_height
        self.font = ImageFont.load_default_imagefont()
        bg, self.ox, self.oy = _house(room)
        self.height, self.width = bg.shape[0] + hud_height, bg.shape[1]
        for y in range(room.height):
            for x in range(room.width):
                if (x, y) in room.obstacles:
                    obs = room.obstacles
                    xs = sorted(ox for ox, oy in obs if oy == y)
                    tile = _table_tile(
                        _floor_tile(x, y),
                        (x - 1, y) not in obs,
                        (x + 1, y) not in obs,
                        (x, y - 1) not in obs,
                        (x, y + 1) not in obs,
                        centerpiece=x == xs[len(xs) // 2],
                    )
                else:
                    tile = _floor_tile(x, y)
                bg[self.oy + y * T : self.oy + (y + 1) * T, self.ox + x * T : self.ox + (x + 1) * T] = tile
        self.background = bg

    def _px(self, pos) -> tuple[int, int]:
        """Top-left pixel of a (possibly fractional) grid position."""
        return self.ox + int(round(pos[0] * T)), self.oy + int(round(pos[1] * T)) + self.hud

    def frame(
        self,
        state: State,
        robot_pos=None,
        dog_pos=None,
        hud: str = "",
        bob: int = 0,
        show_view: bool = True,
        reboot: bool = False,
    ) -> Image.Image:
        env = self.env
        robot_pos = state.robot_pos if robot_pos is None else robot_pos
        dog_pos = state.dog_pos if dog_pos is None else dog_pos
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[: self.hud] = (26, 28, 44)
        canvas[self.hud :] = self.background

        if show_view and not state.dog_playing:
            dog = env.index[state.dog_pos]
            for c in np.flatnonzero(env.sees[dog, state.dog_face]):
                px, py = self._px(env.cells[c])
                region = canvas[py : py + T, px : px + T].astype(float)
                canvas[py : py + T, px : px + T] = (0.7 * region + 0.3 * np.array((255, 120, 190))).astype(np.uint8)

        if state.mess_pos is not None:
            _blit(canvas, SPRITES["pee"], *self._px(state.mess_pos))

        rx, ry = self._px(robot_pos)
        robot = SPRITES["robot"].copy()
        if reboot:
            robot[..., :3] = (robot[..., :3] * 0.5).astype(np.uint8)
        _blit(canvas, robot, rx, ry)

        dx, dy = self._px(dog_pos)
        _blit(canvas, SPRITES["dog"][state.dog_face], dx, dy - bob)
        if state.dog_playing:
            _blit(canvas, SPRITES["bubble"], dx + 2, dy - 12)
        if reboot:
            _blit(canvas, SPRITES["heart"], rx + 9, ry - 4)

        img = Image.fromarray(canvas)
        if hud:
            ImageDraw.Draw(img).text((3, 1), hud, fill=(244, 244, 244), font=self.font)
        return img


def rollout(env: CleaningEnv, policy: PolicyMixin, steps: int, state: State | None = None, seed=None):
    """Simulate `steps` rounds. Returns the lists of states, actions, rewards, events
    and reboot trajectories (see CleaningEnv.step)."""
    rng = np.random.default_rng(seed)
    state = env.random_init(rng) if state is None else state
    states, actions, rewards, events, reboots = [state], [], [], [], []
    for _ in range(steps):
        a = policy.act(state, rng)
        rewards.append(env.reward(state))
        state, event, reboot = env.step(state, a, rng)
        actions.append(a)
        states.append(state)
        events.append(event)
        reboots.append(reboot)
    return states, actions, rewards, events, reboots


def animate(
    policy: PolicyMixin,
    env: CleaningEnv,
    steps: int = 100,
    state: State | None = None,
    seed=None,
    filename: str | None = None,
    tween: int = 4,
    frame_ms: int = 60,
    scale: int = 3,
    show_view: bool = True,
) -> list[Image.Image]:
    """Animate `policy` for `steps` rounds. Saves a GIF to `filename` (if given)
    and returns the frames."""
    renderer = Renderer(env)
    states, actions, rewards, events, reboots = rollout(env, policy, steps, state, seed)
    frames: list[Image.Image] = []
    total, captures = 0.0, 0
    for t in range(steps):
        s0, s1, event = states[t], states[t + 1], events[t]
        total += rewards[t]
        captures += event == "captured"
        hud = f"t={t + 1:3d} R={total:5.0f} k={env.k} caught={captures}"
        if event == "captured":
            # Robot is caught: show the dog reaching it, then the reboot.
            caught_at = _caught_at(env, s0, actions[t])
            chasing = replace(s0, dog_playing=True)
            for i in range(tween):
                f = (i + 1) / tween
                rpos = tuple(a + f * (b - a) for a, b in zip(s0.robot_pos, caught_at))
                dpos = tuple(a + f * (b - a) for a, b in zip(s0.dog_pos, caught_at))
                frames.append(renderer.frame(chasing, robot_pos=rpos, dog_pos=dpos, hud=hud, show_view=show_view))
            # Then the k rounds in which the dog wanders while the robot reboots.
            path = reboots[t]
            for _ in range(tween):
                frames.append(renderer.frame(path[0], hud=f"REBOOT 0/{env.k}", show_view=show_view, reboot=True))
            for j in range(1, len(path)):
                _tween_round(frames, renderer, path[j - 1], path[j], tween, f"REBOOT {j}/{env.k}", show_view, reboot=True)
            continue
        _tween_round(frames, renderer, s0, s1, tween, hud, show_view)

    frames = [f.resize((f.width * scale, f.height * scale), Image.NEAREST) for f in frames]
    if filename:
        frames[0].save(filename, save_all=True, append_images=frames[1:], duration=frame_ms, loop=0)
    return frames


def _tween_round(frames, renderer, s0: State, s1: State, tween: int, hud: str, show_view: bool, reboot=False):
    for i in range(tween):
        f = (i + 1) / tween
        rpos = tuple(a + f * (b - a) for a, b in zip(s0.robot_pos, s1.robot_pos))
        dpos = tuple(a + f * (b - a) for a, b in zip(s0.dog_pos, s1.dog_pos))
        shown = s1 if i == tween - 1 else State(s0.robot_pos, s0.dog_pos, s1.dog_face, s1.dog_playing, s0.mess_pos)
        bob = 1 if (i % 2 and s0.dog_pos != s1.dog_pos) else 0
        frames.append(
            renderer.frame(shown, robot_pos=rpos, dog_pos=dpos, hud=hud, bob=bob, show_view=show_view, reboot=reboot)
        )


def _caught_at(env: CleaningEnv, state: State, action: int):
    """Where the robot was when the dog reached it (before rebooting)."""
    if action < 4:
        x, y = state.robot_pos[0] + DIRS[action][0], state.robot_pos[1] + DIRS[action][1]
        if env.room.is_free((x, y)):
            return (x, y)
    return state.robot_pos
