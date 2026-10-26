from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
import random
from typing import Any, Iterable

Cell = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class WorldConfig:
    grid_shape: Cell
    max_steps: int
    seed: int
    neighborhood_radius: int
    checkpoint_every: int
    particle_counts: dict[str, int]
    interaction_weights: dict[str, int]

    @property
    def volume(self) -> int:
        x, y, z = self.grid_shape
        return x * y * z


@dataclass(slots=True)
class WorldState:
    positions: list[Cell]
    types: list[str]
    occupied: dict[Cell, int]
    step: int = 0

    def clone(self) -> "WorldState":
        return WorldState(
            positions=list(self.positions),
            types=list(self.types),
            occupied=dict(self.occupied),
            step=self.step,
        )


def config_from_dict(data: dict[str, Any]) -> WorldConfig:
    world = data.get("world", {})
    particle_counts = {str(k): int(v) for k, v in data.get("particle_counts", {}).items()}
    if not particle_counts:
        raise ValueError("Config must contain non-empty particle_counts.")

    interaction_weights = {
        str(k): int(v)
        for k, v in data.get("interaction_weights", {}).items()
    }
    for particle_type in particle_counts:
        interaction_weights.setdefault(particle_type, 1)

    grid_shape_raw = world.get("grid_shape", [16, 16, 16])
    if len(grid_shape_raw) != 3:
        raise ValueError("world.grid_shape must have exactly 3 dimensions.")
    grid_shape = tuple(int(v) for v in grid_shape_raw)
    if any(v <= 0 for v in grid_shape):
        raise ValueError("world.grid_shape dimensions must be positive.")

    cfg = WorldConfig(
        grid_shape=grid_shape,  # type: ignore[arg-type]
        max_steps=int(world.get("max_steps", 50)),
        seed=int(world.get("seed", 42)),
        neighborhood_radius=int(world.get("neighborhood_radius", 2)),
        checkpoint_every=int(world.get("checkpoint_every", 10)),
        particle_counts=particle_counts,
        interaction_weights=interaction_weights,
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: WorldConfig) -> None:
    if cfg.max_steps < 0:
        raise ValueError("max_steps must be non-negative.")
    if cfg.neighborhood_radius < 1:
        raise ValueError("neighborhood_radius must be >= 1.")
    if cfg.checkpoint_every < 1:
        raise ValueError("checkpoint_every must be >= 1.")
    if sum(cfg.particle_counts.values()) > cfg.volume:
        raise ValueError("Particle count exceeds grid volume.")
    if sum(cfg.particle_counts.values()) > 10_000:
        raise ValueError("MVP limit: particles <= 10000.")
    if any(v > 64 for v in cfg.grid_shape):
        raise ValueError("MVP limit: grid dimension <= 64.")
    if len(cfg.particle_counts) > 4:
        raise ValueError("MVP limit: particle types <= 4.")
    if cfg.max_steps > 5000:
        raise ValueError("MVP limit: max_steps <= 5000.")
    if any(count < 0 for count in cfg.particle_counts.values()):
        raise ValueError("Particle counts must be non-negative.")
    if any(weight < 0 for weight in cfg.interaction_weights.values()):
        raise ValueError("Interaction weights must be non-negative integers.")


def init_world(cfg: WorldConfig) -> WorldState:
    rng = random.Random(cfg.seed)
    positions: list[Cell] = []
    types: list[str] = []
    occupied: dict[Cell, int] = {}

    for particle_type in sorted(cfg.particle_counts):
        for _ in range(cfg.particle_counts[particle_type]):
            pos = random_empty_cell(rng, cfg.grid_shape, occupied)
            occupied[pos] = len(positions)
            positions.append(pos)
            types.append(particle_type)

    return WorldState(positions=positions, types=types, occupied=occupied)


def random_empty_cell(rng: random.Random, shape: Cell, occupied: dict[Cell, int]) -> Cell:
    while True:
        cell = (
            rng.randrange(shape[0]),
            rng.randrange(shape[1]),
            rng.randrange(shape[2]),
        )
        if cell not in occupied:
            return cell


def neighbors(cell: Cell, shape: Cell, include_self: bool = True) -> list[Cell]:
    x, y, z = cell
    deltas = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ]
    if include_self:
        deltas.append((0, 0, 0))

    out: list[Cell] = []
    for dx, dy, dz in deltas:
        nx, ny, nz = x + dx, y + dy, z + dz
        if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
            out.append((nx, ny, nz))
    return out


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def local_integer_potential(candidate: Cell, particle_index: int, state: WorldState, cfg: WorldConfig) -> int:
    particle_type = state.types[particle_index]
    radius = cfg.neighborhood_radius
    weight = cfg.interaction_weights.get(particle_type, 1)
    pot = 0
    cx, cy, cz = candidate
    sx, sy, sz = cfg.grid_shape

    for x in range(max(0, cx - radius), min(sx, cx + radius + 1)):
        for y in range(max(0, cy - radius), min(sy, cy + radius + 1)):
            for z in range(max(0, cz - radius), min(sz, cz + radius + 1)):
                idx = state.occupied.get((x, y, z))
                if idx is None or idx == particle_index:
                    continue
                if state.types[idx] != particle_type:
                    continue
                distance = manhattan(candidate, (x, y, z))
                contribution = max(0, radius + 1 - distance)
                pot += weight * contribution

    return pot


def choose_best_neighbor(particle_index: int, state: WorldState, cfg: WorldConfig) -> Cell:
    current = state.positions[particle_index]
    best_cell = current
    best_potential = local_integer_potential(current, particle_index, state, cfg)

    for cell in neighbors(current, cfg.grid_shape):
        occupant = state.occupied.get(cell)
        if occupant is not None and occupant != particle_index:
            continue
        potential = local_integer_potential(cell, particle_index, state, cfg)
        if potential < best_potential or (potential == best_potential and cell < best_cell):
            best_cell = cell
            best_potential = potential

    return best_cell


def step_world_exact(state: WorldState, cfg: WorldConfig) -> WorldState:
    proposals = [choose_best_neighbor(i, state, cfg) for i in range(len(state.positions))]
    winners: dict[Cell, int] = {}

    for i, cell in enumerate(proposals):
        previous = winners.get(cell)
        if previous is None or i < previous:
            winners[cell] = i

    new_positions: list[Cell] = []
    new_occupied: dict[Cell, int] = {}
    for i, old_cell in enumerate(state.positions):
        proposed = proposals[i]
        chosen = proposed if winners.get(proposed) == i else old_cell
        if chosen in new_occupied:
            chosen = old_cell
        new_occupied[chosen] = i
        new_positions.append(chosen)

    return WorldState(
        positions=new_positions,
        types=list(state.types),
        occupied=new_occupied,
        step=state.step + 1,
    )


def state_hash(state: WorldState) -> str:
    h = blake2b(digest_size=16)
    h.update(state.step.to_bytes(8, "little", signed=False))
    for idx, (cell, particle_type) in enumerate(zip(state.positions, state.types)):
        h.update(idx.to_bytes(8, "little", signed=False))
        h.update(particle_type.encode("utf-8"))
        for coord in cell:
            h.update(coord.to_bytes(4, "little", signed=False))
    return h.hexdigest()


def run_world(cfg: WorldConfig) -> Iterable[WorldState]:
    state = init_world(cfg)
    yield state
    for _ in range(cfg.max_steps):
        state = step_world_exact(state, cfg)
        yield state


def state_to_dict(state: WorldState) -> dict[str, Any]:
    type_counts: dict[str, int] = {}
    for particle_type in state.types:
        type_counts[particle_type] = type_counts.get(particle_type, 0) + 1
    return {
        "step": state.step,
        "state_hash": state_hash(state),
        "n_particles": len(state.positions),
        "type_counts": dict(sorted(type_counts.items())),
        "positions": [list(cell) for cell in state.positions],
        "types": list(state.types),
    }


def state_from_dict(data: dict[str, Any]) -> WorldState:
    positions = [tuple(int(v) for v in cell) for cell in data["positions"]]
    types = [str(v) for v in data["types"]]
    occupied = {cell: idx for idx, cell in enumerate(positions)}
    return WorldState(
        positions=positions,  # type: ignore[arg-type]
        types=types,
        occupied=occupied,
        step=int(data.get("step", 0)),
    )
