'''Structure of arrays container for holding numpy arrays that correspond to pseudojets'''

import numpy as np

from numba import njit

from pyantikt.pseudojet import PseudoJet

class NPPseudoJets:
    def __init__(self, size:int):
        '''Setup blank arrays that will be filled later'''
        self.size = size
        self.phi = np.zeros(size, dtype=float)              # phi
        self.rap = np.zeros(size, dtype=float)              # rapidity
        self.inv_pt2 = np.zeros(size, dtype=float)          # 1/pt^2
        self.dist = np.zeros(size, dtype=float)             # nearest neighbour geometric distance
        self.akt_dist = np.zeros(size, dtype=float)         # nearest neighbour antikt metric
        self.nn = np.zeros(size, dtype=int)                 # index of my nearest neighbour
        self.mask = np.ones(size, dtype=bool)               # if True this is not an active jet anymore
        self.jets_index = np.zeros(size, dtype=int)         # index reference to the PseudoJet list
    
    def set_jets(self, jets:list[PseudoJet]):
        if len(jets) > self.phi.size:
            raise RuntimeError(f"Attempted to fill NP PseudoJets, but containers are too small ({self.size})")
        for ijet, jet in enumerate(jets):
            self.phi[ijet] = jet.phi
            self.rap[ijet] = jet.rap
            self.inv_pt2[ijet] = jet.inv_pt2
            self.nn[ijet] = -1
            self.dist[ijet] = self.akt_dist[ijet] = 1e20
            self.mask[ijet] = False
            self.jets_index[ijet] = ijet
        self.next_slot = len(jets)
        self.dist[len(jets):] = self.akt_dist[len(jets):] = 1e20

    def __str__(self) -> str:
        _string = ""
        for ijet in range(self.phi.size):
            _string += (f"{ijet} - {self.phi[ijet]} {self.rap[ijet]} {self.inv_pt2[ijet]} {self.dist[ijet]} "
            f"{self.akt_dist[ijet]} {self.nn[ijet]} {self.jets_index[ijet]} "
            f"(mask: {self.mask[ijet]})\n")
        return _string
    
    def mask_slot(self, ijet: int):
        self.mask[ijet] = True
        self.dist[ijet] = self.akt_dist[ijet] = 1e20

    def insert_jet(self, jet: PseudoJet, slot:int, jet_index: int):
        '''Add a new pseudojet into the numpy structures'''
        if slot >= self.size:
            raise RuntimeError(f"Attempted to fill a jet into a slot that doesn't exist (slot {slot} >= size {self.size})")
        self.phi[slot] = jet.phi
        self.rap[slot] = jet.rap
        self.inv_pt2[slot] = jet.inv_pt2
        self.nn[slot] = -1
        self.dist[slot] = self.akt_dist[slot] = 1e20 # Necessary?
        self.jets_index[slot] = jet_index
        self.mask[slot] = False
    
    def print_jet(self, ijet:int) -> str:
        return (f"{ijet} - {self.phi[ijet]} {self.rap[ijet]} {self.inv_pt2[ijet]} "
        f"{self.dist[ijet]} {self.akt_dist[ijet]} {self.nn[ijet]} {self.jets_index[ijet]} "
        f"(mask: {self.mask[ijet]} -> {self.mask[self.nn[ijet]] if self.nn[ijet] >= 0 else None})")
    