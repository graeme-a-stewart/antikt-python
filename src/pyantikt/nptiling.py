# Class definitions for tiling and tiled jets, numpy version

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numba import njit

from pyantikt.nppseudojet import NPPseudoJets

# This one is so simple, just keep it as a dataclass
@dataclass
class TilingDef:
    """Simple class with tiling parameters"""

    tiles_rap_min: float    # Minimum rapidity we have
    tiles_rap_max: float    # Maximum rapidity we have
    tile_size_rap: float    # Size of each rapidity tile (usually R^2)
    tile_size_phi: float    # Size of each phi tile      (2\pi / R^2)
    n_tiles_rap: int        # Number of rapidity tiles
    n_tiles_phi: int        # Number of phi tiles
    tiles_irap_min: int     # min_rapidity / tile_size
    tiles_irap_max: int     # max_rapidity / tile_size

    def __post_init__(self):
        self.n_tiles = self.n_tiles_rap * self.n_tiles_phi

class NPTiling:
    """Hold the tiling data structure, which is 3D of 
    [rap/eta tile index][phi tile index][pseudojet]"""
    def __init__(self, setup:TilingDef, max_jets_per_tile:int):
        self.setup = setup
        self.max_jets_per_tile = max_jets_per_tile
        self.phi = np.zeros((setup.n_tiles_rap, setup.n_tiles_phi, max_jets_per_tile), dtype=float)         # phi
        self.rap = np.zeros((setup.n_tiles_rap, setup.n_tiles_phi, max_jets_per_tile), dtype=float)         # rapidity
        self.inv_pt2 = np.zeros((setup.n_tiles_rap, setup.n_tiles_phi, max_jets_per_tile), dtype=float)     # 1/pt^2
        self.dist = np.zeros((setup.n_tiles_rap, setup.n_tiles_phi, max_jets_per_tile), dtype=float)        # nearest neighbour geometric distance
        self.nn = np.zeros((setup.n_tiles_rap, setup.n_tiles_phi, max_jets_per_tile), dtype=int)            # nearest neighbour antikt metric
        self.mask = np.ones((setup.n_tiles_rap, setup.n_tiles_phi, max_jets_per_tile), dtype=bool)          # if True this is not an active jet slot
        self.npjets_index = np.zeros((setup.n_tiles_rap, setup.n_tiles_phi, max_jets_per_tile), dtype=int)  # index reference to the NPPseudoJet


        # self.size = size
        # self.phi = np.zeros(size, dtype=float)              # phi
        # self.rap = np.zeros(size, dtype=float)              # rapidity
        # self.inv_pt2 = np.zeros(size, dtype=float)          # 1/pt^2
        # self.dist = np.zeros(size, dtype=float)             # nearest neighbour geometric distance
        # self.akt_dist = np.zeros(size, dtype=float)         # nearest neighbour antikt metric
        # self.nn = np.zeros(size, dtype=int)                 # index of my nearest neighbour
        # self.mask = np.ones(size, dtype=bool)               # if True this is not an active jet anymore
        # self.jets_index = np.zeros(size, dtype=int)         # index reference to the PseudoJet list

    def fill_with_jets(self, npjets:NPPseudoJets):
        # First bulk calculate the bin indexes we need to use
        _irap, _iphi = _get_tile_indexes(self.setup.tiles_rap_min, 
                                       self.setup.tile_size_rap,
                                       self.setup.n_tiles_rap,
                                       self.setup.n_tiles_phi,
                                       npjets.rap,
                                       npjets.phi)
        for r, p in zip(_irap, _iphi):
            print(f"({r}, {p})")
    
@njit
def _get_tile_indexes(tiles_rap_min:npt.DTypeLike, 
                        tile_size_rap:npt.DTypeLike,
                        n_tiles_rap:npt.DTypeLike,
                        n_tiles_phi:npt.DTypeLike,
                        rap:npt.ArrayLike,
                        phi:npt.ArrayLike):
    # print(np.min(rap), np.max(rap))
    _irap = (rap - tiles_rap_min) / tile_size_rap
    # print(_irap)
    _irap = _irap.astype(np.int64)
    _irap = np.where(_irap < 0, 0, _irap)
    _irap = np.where(_irap >= n_tiles_rap, n_tiles_rap-1, _irap)
    print(_irap)

    _iphi = phi * n_tiles_phi / (2.0 * np.pi)
    _iphi = _iphi.astype(np.int64)
    print(_iphi)

    return _irap, _iphi
