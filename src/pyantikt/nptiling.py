# Class definitions for tiling and tiled jets, numpy version

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numba import njit

from pyantikt.nppseudojet import NPPseudoJets
from pyantikt.pseudojet import PseudoJet

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
        self.rap = np.zeros((setup.n_tiles_rap, setup.n_tiles_phi, max_jets_per_tile), dtype=float)         # rapidity
        self.phi = np.zeros((setup.n_tiles_rap, setup.n_tiles_phi, max_jets_per_tile), dtype=float)         # phi
        self.inv_pt2 = np.zeros((setup.n_tiles_rap, setup.n_tiles_phi, max_jets_per_tile), dtype=float)     # 1/pt^2
        self.dist = np.zeros((setup.n_tiles_rap, setup.n_tiles_phi, max_jets_per_tile), dtype=float)        # nearest neighbour geometric distance
        self.akt_dist = np.zeros((setup.n_tiles_rap, setup.n_tiles_phi, max_jets_per_tile), dtype=float)    # nearest neighbour antikt metric
        self.nn = np.zeros((setup.n_tiles_rap, setup.n_tiles_phi, max_jets_per_tile), dtype=(np.int64,3))        # nearest neighbour coordinates
        self.mask = np.ones((setup.n_tiles_rap, setup.n_tiles_phi, max_jets_per_tile), dtype=bool)          # if True this is not an active jet slot
        self.jets_index = np.zeros((setup.n_tiles_rap, setup.n_tiles_phi, max_jets_per_tile), dtype=int)    # index reference to the PseudoJet

        # Setup safety values for distances
        self.dist.fill(1e20)
        self.akt_dist.fill(1e20)
        self.nn[0].fill(-1)
        self.nn[1].fill(-1)
        self.nn[2].fill(-1)

        # This tuple holds the rightmost neighbour tiles of any tile
        # N.B. phi wraps, but rap does not
        # If an entry is (-1,-1) it's an invalid neighbour
        # The 4 members of this array are up, up-right, right, down-right:
        #      -  0  1
        #      -  X  2
        #      -  -  3
        self.righttiles = np.zeros((setup.n_tiles_rap, setup.n_tiles_phi, 4), dtype=(np.int64, 2))
        for irap in range(setup.n_tiles_rap):
            for iphi in range(setup.n_tiles_phi):
                if irap == 0:
                    self.righttiles[irap,iphi,0] = (-1,-1)
                    self.righttiles[irap,iphi,1] = (-1,-1)
                else:
                    self.righttiles[irap,iphi,0] = (irap-1,iphi)
                    self.righttiles[irap,iphi,1] = (irap-1,iphi+1 if iphi != setup.n_tiles_phi-1 else 0)
                self.righttiles[irap,iphi,2] = (irap,iphi+1 if iphi != setup.n_tiles_phi-1 else 0)
                if irap == setup.n_tiles_rap-1:
                    self.righttiles[irap,iphi,3] = (-1,-1)
                else:
                    self.righttiles[irap,iphi,3] = (irap+1,iphi+1 if iphi != setup.n_tiles_phi-1 else 0)

        # This tuple holds all the neighbour tiles of any tile
        # N.B. phi wraps, but rap does not
        # If an entry is (-1,-1) it's an invalid neighbour
        # The 8 members of this array scan increasing phi and increasing rapidity
        #      0  1  2
        #      3  X  4
        #      5  6  7
        self.neighbourtiles = np.zeros((setup.n_tiles_rap, setup.n_tiles_phi, 8), dtype=(np.int64, 2))
        for irap in range(setup.n_tiles_rap):
            for iphi in range(setup.n_tiles_phi):
                self.neighbourtiles[irap,iphi,0] = (irap-1,iphi-1)
                self.neighbourtiles[irap,iphi,1] = (irap-1,iphi)
                self.neighbourtiles[irap,iphi,2] = (irap-1,iphi+1)
                self.neighbourtiles[irap,iphi,3] = (irap  ,iphi-1)
                self.neighbourtiles[irap,iphi,4] = (irap  ,iphi+1)
                self.neighbourtiles[irap,iphi,5] = (irap+1,iphi-1)
                self.neighbourtiles[irap,iphi,6] = (irap+1,iphi)
                self.neighbourtiles[irap,iphi,7] = (irap+1,iphi+1)
                for neighbour in self.neighbourtiles[irap,iphi]:
                    if neighbour[0] < 0:
                        neighbour[1] = -1
                    elif neighbour[0] > setup.n_tiles_rap-1:
                        neighbour[0] = neighbour[1] = -1
                    elif neighbour[1] < 0:
                        neighbour[1] = setup.n_tiles_phi-1
                    elif neighbour[1] > setup.n_tiles_phi-1:
                        neighbour[1] = 0
        #         print(f"{irap},{iphi} {self.neighbourtiles[irap,iphi]}")
        # exit(0)

    def fill_with_jets(self, jets:list[PseudoJet], rap, phi):
        # First bulk calculate the bin indexes we need to use
        _irap, _iphi = get_tile_indexes(self.setup.tiles_rap_min, 
                                       self.setup.tile_size_rap,
                                       self.setup.n_tiles_rap,
                                       self.setup.n_tiles_phi,
                                       rap, phi)

        # Now place jets into the correct locations
        # There's no way to avoid a loop here that I can see!
        for ijet, jet in enumerate(jets):
            # Find the first empty slot in the tile
            try:
                islot = np.where(self.mask[_irap[ijet], _iphi[ijet]])[0][0]
            except IndexError:
                print(f"No free tile slot at ({_irap[ijet]}, {_iphi[ijet]})")
                print(ijet, _irap[ijet], _iphi[ijet], self.mask[_irap[ijet], _iphi[ijet]])
            self.rap[_irap[ijet], _iphi[ijet], islot] = jet.rap
            self.phi[_irap[ijet], _iphi[ijet], islot] = jet.phi
            self.inv_pt2[_irap[ijet], _iphi[ijet], islot] = jet.inv_pt2
            self.dist[_irap[ijet], _iphi[ijet], islot] = 1e20
            self.akt_dist[_irap[ijet], _iphi[ijet], islot] = 1e20
            self.nn[_irap[ijet], _iphi[ijet], islot] = (-1,-1,-1)
            self.mask[_irap[ijet], _iphi[ijet], islot] = False
            self.jets_index[_irap[ijet], _iphi[ijet], islot] = ijet
            # print(f"Set jet {ijet} into ({_irap[ijet]}, {_iphi[ijet]}, {islot})")

    def mask_slot(self, ijet:tuple[int]):
        self.mask[ijet] = True
        self.dist[ijet] = self.akt_dist[ijet] = 1e20
        self.nn[ijet] = [-1,-1,-1]
        self.jets_index[ijet] = -1

    def insert_jet(self, newjet:PseudoJet, npjet_index:int):
        """Add a new PseudoJet object into the tiling structure"""
        _irap = int((newjet.rap - self.setup.tiles_rap_min) / self.setup.tile_size_rap)
        if _irap < 0:
            _irap = 0
        elif _irap > self.setup.n_tiles_rap-1:
            _irap = self.setup.n_tiles_rap-1
        _iphi = int(newjet.phi * self.setup.n_tiles_phi / (2.0 * np.pi))
        try:
            islot = np.where(self.mask[_irap, _iphi])[0][0]
        except IndexError:
            print(f"No free tile slot at ({_irap}, {_iphi})")
            print(_irap, _iphi, self.mask[_irap, _iphi])
        self.rap[_irap, _iphi, islot] = newjet.rap
        self.phi[_irap, _iphi, islot] = newjet.phi
        self.inv_pt2[_irap, _iphi, islot] = newjet.inv_pt2
        self.dist[_irap, _iphi, islot] = 1e20
        self.akt_dist[_irap, _iphi, islot] = 1e20
        self.nn[_irap, _iphi, islot] = [-1,-1,-1]
        self.mask[_irap, _iphi, islot] = False
        self.jets_index[_irap, _iphi, islot] = npjet_index
        # print(f"Added new jet {npjet_index} to ({_irap}, {_iphi}, {islot})")
        return _irap, _iphi, islot
    
    def dump_jet(self, ijet:tuple[int]):
        print(f"NPTiledJet {ijet}: {self.rap[ijet]} {self.phi[ijet]} {self.inv_pt2[ijet]} {self.dist[ijet]} "
              f"{self.akt_dist[ijet]} {self.nn[ijet]} {self.mask[ijet]} {self.jets_index[ijet]}")

@njit
def get_tile_indexes(tiles_rap_min:npt.DTypeLike, 
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
    # print(_irap)

    _iphi = phi * n_tiles_phi / (2.0 * np.pi)
    _iphi = _iphi.astype(np.int64)
    # print(_iphi)

    return _irap, _iphi
