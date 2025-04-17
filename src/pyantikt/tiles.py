# Class definitions for tiling and tiled jets

import typing
from dataclasses import dataclass


@dataclass
class TiledJet:
    """Class holding a jet with extra parameters to track tiling
    - note that the default values give an invalid TiledJet
    - in this case original code used
        NN=previous=next=self (i.e., the same pointer as *this in C++)
        but None is much more Pythonic
    """

    id: int = -1
    eta: float = 0.0
    phi: float = 0.0
    kt2: float = 0.0
    NN_dist: float = 0.0
    diJ_dist: float = 0.0
    jet_index: int = -1
    tile_index: tuple[int, int] | None = (-1, -1)
    diJ_posn: int = -1
    # Cannot use a TiledJet type here as the class isn't yet defined
    # This is a work around, allowing any valid object to be used
    # Note that these are references to the actual TiledJet objects, but are set to
    # None if they are invalid.
    NN: "typing.Any" = None
    previous: "typing.Any" = None
    next: "typing.Any" = None

    # Keep a marker here is a jet is still active or not (i.e. has been merged)
    active: bool = True

    def isvalid(self):
        return False if self.id == -1 else True

    def __iter__(self):
        return self

    def __next__(self):
        if self.next:
            return self.next
        raise StopIteration


@dataclass
class TilingDef:
    """Simple class with tiling parameters"""

    tiles_eta_min: float
    tiles_eta_max: float
    tile_size_eta: float
    tile_size_phi: float
    n_tiles_eta: int
    n_tiles_phi: int
    tiles_ieta_min: float
    tiles_ieta_max: float

    def __post_init__(self):
        self.n_tiles = self.n_tiles_eta * self.n_tiles_phi


@dataclass
class Tiling:
    setup: TilingDef
    tiles: list[list[TiledJet]]
    positions: list[list[int]]
    tags: list[list[bool]]

    def __init__(self, setup: TilingDef):
        self.setup = setup
        # self.tiled should store a list of TiledJet objects, which starts empty for now
        self.tiles = [
            [list() for i in range(setup.n_tiles_phi)] for j in range(setup.n_tiles_eta)
        ]
        # # TBD
        # self.positions = [
        #     [0 for i in range(setup.n_tiles_phi)] for j in range(setup.n_tiles_eta)
        # ]
        # # Tag tiles marked for an update (not needed if update is managed with a set())
        # self.tags = [
        #     [False for i in range(setup.n_tiles_phi)] for j in range(setup.n_tiles_eta)
        # ]


def rightneighbours(row: int, column: int, setup: TilingDef) -> list[tuple[int, int]]:
    """Return all of the valid tile indexes to the right of the one at (row, column)
    The neighbourhood of a tile is set up as follows

        LRR
        LXR
        LLR

    Appropriate precautions are taken at the edges
    Note that phi wraps round, but not eta
    """
    rightneighbours = []
    if row != 0:
        rightneighbours.append((row - 1, column))
        if column != setup.n_tiles_phi - 1:
            rightneighbours.append((row - 1, column + 1))
        else:
            # Wrap around
            rightneighbours.append((row - 1, 0))
    if column != setup.n_tiles_phi - 1:
        rightneighbours.append((row, column + 1))
    else:
        rightneighbours.append((row, 0))
    if row != setup.n_tiles_eta - 1:
        if column != setup.n_tiles_phi - 1:
            rightneighbours.append((row + 1, column + 1))
        else:
            # Wrap around
            rightneighbours.append((row + 1, 0))

    # print(f"Start: {row}, {column}")
    # for n in rightneighbours:
    #     print(n)
    return rightneighbours


def surrounding_tiles(tile_index: tuple[int, int], tiling: Tiling):
    """Return the set of valid surrounding tiles for a particular tile index"""
    surrounding_tiles = set()
    for delta_row in range(-1, 2):
        for delta_column in range(-1, 2):
            new_tile = [tile_index[0] + delta_row, tile_index[1] + delta_column]
            if new_tile[0] >= 0 and new_tile[0] < tiling.setup.n_tiles_eta:
                if new_tile[1] < 0:
                    new_tile[1] = tiling.setup.n_tiles_phi - 1
                elif new_tile[1] >= tiling.setup.n_tiles_phi:
                    new_tile[1] = 0
                surrounding_tiles.add(tuple(new_tile))

    return surrounding_tiles


def add_untagged_neighbours_to_tile_union(
    center_index: tuple[int, int], tile_union: set, tiling: Tiling
):
    """Adds to the set tile_union the tiles that are in the neighbourhood
    of the specified tile_index, including itself and whose tagged status are
    false - start adding from position n_near_tiles-1, and increase n_near_tiles as
    you go along. When a neighbour is added its tagged status is set to true (NOT USED).
    """

    tile_union.update(surrounding_tiles(center_index, tiling))
    # print(tile_union, len(tile_union))

    return len(tile_union)
