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
    jet_index: int = -1
    tile_index: tuple[int, int] | None = (-1, -1)
    diJ_posn: int = -1
    # Cannot use a TiledJet type here as the class isn't yet defined
    # This is a work around, allowing any valid object to be used
    NN: "typing.Any" = None
    previous: "typing.Any" = None
    next: "typing.Any" = None

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
    tile_size_eta: int
    tile_size_phi: int
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
            [ list() for i in range(setup.n_tiles_phi)] for j in range(setup.n_tiles_eta)
        ]
        # TBD (2D?)
        self.positions = [
            [0 for i in range(setup.n_tiles_phi)] for j in range(setup.n_tiles_eta)
        ]
        # TBD (2D?)
        self.tags = [
            [False for i in range(setup.n_tiles_phi)] for j in range(setup.n_tiles_eta)
        ]
