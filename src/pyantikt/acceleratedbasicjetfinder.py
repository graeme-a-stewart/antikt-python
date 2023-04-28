
from copy import deepcopy
from dataclasses import dataclass
from math import pi
from pyantikt.history import HistoryElement, ClusterSequence, initial_history, HistoryState
from pyantikt.pseudojet import PseudoJet
from pyantikt.nppseudojet import NPPseudoJets
from sys import float_info

import logging
import numpy as np
import numpy.typing as npt
import numpy.ma as ma

logger = logging.getLogger("jetfinder")

from numba import njit

Invalid = -3
NonexistentParent = -2
BeamJet = -1

@dataclass
class BasicJetInfo:
    '''This is an add on class where we can augment the pseudojets
    with nearest neighbour information'''
    id: int = -1          # My jetID (which will be the index in the jet list)
    nn: int | None = None # ID of my nearest neighbour
    nn_dist: float = float_info.max  # Geometric distance
    akt_dist: float = float_info.max # AntiKt jet distance
    active: bool = True

@njit
def geometric_distance(jetAphi: float, jetArap: float, jetBphi: float, jetBrap:float):
    '''Distance between two jets'''
    dphi = pi - abs(pi - abs(jetAphi - jetBphi))
    drap = jetArap - jetBrap
    return dphi * dphi + drap * drap

@njit
def antikt_distance(jetAdist: float, jetAipt2: float, jetBipt2:float):
    '''AntiKt distance between two jets'''
    if jetBipt2 > 0.0:
        inv_pt2 = min(jetAipt2, jetBipt2)
    else:
        inv_pt2 = jetAipt2
    return jetAdist * inv_pt2

@njit
def find_closest_jets(akt_dist:npt.ArrayLike, nn:npt.ArrayLike):
    '''Look over active jets and find the closest'''
    closest = akt_dist.argmin()
    return akt_dist[closest], closest


@njit
def scan_for_all_nearest_neighbours(phi: npt.ArrayLike, rap: npt.ArrayLike, inv_pt2: npt.ArrayLike,
                                    dist: npt.ArrayLike, akt_dist: npt.ArrayLike,
                                    nn: npt.ArrayLike, mask:npt.ArrayLike, R2:float):
    '''Do a full scan for nearest (geometrical) neighbours'''
    for ijet in range(phi.size):
        if mask[ijet]:
            continue
        _dphi = np.pi - np.abs(np.pi - np.abs(phi - phi[ijet]))
        _drap = rap - rap[ijet]
        _dist = _dphi*_dphi + _drap*_drap
        _dist[ijet] = R2 # Avoid measuring the distance 0 to myself!
        _dist[mask] = 1e20 # Don't consider any masked jets
        iclosejet = _dist.argmin()
        dist[ijet] = _dist[iclosejet]
        if iclosejet == ijet:
            nn[ijet] = -1
            akt_dist[ijet] = dist[ijet] * inv_pt2[ijet]
        else:
            nn[ijet] = iclosejet
            akt_dist[ijet] = dist[ijet] * (inv_pt2[ijet] if inv_pt2[ijet] < inv_pt2[iclosejet] else inv_pt2[iclosejet])

@njit
def scan_for_my_nearest_neighbours(ijet:int, phi: npt.ArrayLike, 
                                   rap:npt.ArrayLike, inv_pt2:npt.ArrayLike, 
                                   dist:npt.ArrayLike, akt_dist:npt.ArrayLike, nn:npt.ArrayLike, 
                                   mask:npt.ArrayLike, R2: float):
    '''Retest all other jets against the target jet'''
    nn[ijet] = -1
    dist[ijet] = R2
    _dphi = np.pi - np.abs(np.pi - np.abs(phi - phi[ijet]))
    _drap = rap - rap[ijet]
    _dist = _dphi*_dphi + _drap*_drap
    _dist[ijet] = R2 # Avoid measuring the distance 0 to myself!
    _dist[mask] = 1e20 # Don't consider any masked jets
    iclosejet = _dist.argmin()
    dist[ijet] = _dist[iclosejet]
    if iclosejet == ijet:
        nn[ijet] = -1
        akt_dist[ijet] = dist[ijet] * inv_pt2[ijet]
    else:
        nn[ijet] = iclosejet
        akt_dist[ijet] = dist[ijet] * (inv_pt2[ijet] if inv_pt2[ijet] < inv_pt2[iclosejet] else inv_pt2[iclosejet])
        # As this function is called on new PseudoJets it's possible
        # that we are now the NN of our NN
        if dist[iclosejet] > dist[ijet]:
            dist[iclosejet] = dist[ijet]
            nn[iclosejet] = ijet
            akt_dist[iclosejet] = dist[iclosejet] * (inv_pt2[ijet] if inv_pt2[ijet] < inv_pt2[iclosejet] else inv_pt2[iclosejet])
 

def test_for_nearest_neighbour(jetA: PseudoJet, jetB: PseudoJet):
    '''Test two jets and see if they are nearest neighbours'''
    if jetA.info.id == jetB.info.id:
        return
    dist = geometric_distance(jetA.phi, jetA.rap, jetB.phi, jetB.rap)
    if dist < jetA.info.nn_dist:
        jetA.info.nn_dist = dist
        jetA.info.nn = jetB.info.id
        jetA.info.akt_dist = antikt_distance(jetA.info.nn_dist, jetA.inv_pt2, jetB.inv_pt2)
    if dist < jetB.info.nn_dist:
        jetB.info.nn_dist = dist
        jetB.info.nn = jetA.info.id
        jetB.info.akt_dist = antikt_distance(jetB.info.nn_dist, jetB.inv_pt2, jetA.inv_pt2)


def compare_status(working:NPPseudoJets, test:NPPseudoJets):
    '''Test two different copies of numpy pseudojet containers that should be equal'''
    dist_diff = working.akt_dist!=test.akt_dist
    idist_diff = np.where(dist_diff)
    if len(idist_diff[0]) > 0:
        print(f"Differences found after full scan of NNs: {idist_diff[0]}")
        for ijet in idist_diff[0]:
            print(f"{ijet}\nW: {working.print_jet(ijet)}\nT: {test.print_jet(ijet)}")
        raise RuntimeError("Jet sets are not the same and they should be!")


def add_step_to_history(history: list[HistoryElement], jets: list[PseudoJet], 
                        parent1: int, parent2: int, jetp_index: int, distance: float):
    '''Add a merging step to the history of clustering
        history - list of HistoryElement entities
        jets - list of pseudojets
        parent1 - the *history* element which is the parent of this merger
        parent2 - the *history* element which is the parent of this merger (can be Invalid)
        jetp_index - the new pseudojet that results from this merger (if both parents exist)
        distance - the distance metric for this merge step
    '''
    max_dij_so_far = max(distance, history[-1].max_dij_so_far)

    history.append(HistoryElement(parent1=parent1, parent2=parent2,
                                    jetp_index=jetp_index, dij=distance,
                                    max_dij_so_far=max_dij_so_far))
    local_step = len(history) - 1
    # logger.debug(f"Added history step {local_step}: {history[-1]}")

    if parent1 >= 0:
        if history[parent1].child != -1:
            raise (
                RuntimeError(
                    f"Internal error. Trying to recombine a parent1 object that has previsously been recombined: {parent1}"
                )
            )
    history[parent1].child = local_step

    if parent2 >= 0:
        if history[parent2].child != -1:
            raise (
                RuntimeError(
                    f"Internal error. Trying to recombine a parent1 object that has previsously been recombined: {parent2}"
                )
            )
    history[parent2].child = local_step

    # get cross-referencing right from PseudoJets
    if jetp_index >= 0:
        jets[jetp_index].cluster_hist_index = local_step

def inclusive_jets(jets: list[PseudoJet], history: list[HistoryElement], ptmin:float=0.0):
    '''return all inclusive jets of a ClusterSequence with pt > ptmin'''
    dcut = ptmin * ptmin
    jets_local = list()
    # For inclusive jets with a plugin algorithm, we make no
    # assumptions about anything (relation of dij to momenta,
    # ordering of the dij, etc.)
    for elt in reversed(history):
        if elt.parent2 != BeamJet:
            continue
        iparent_jet = history[elt.parent1].jetp_index
        jet = jets[iparent_jet]
        if jet.pt2 >= dcut:
            jets_local.append(jet)

    return jets_local

def basicjetfinder(initial_particles: list[PseudoJet], Rparam: float=0.4, ptmin: float=0.0):
    """Basic AntiKt Jet finding code"""
    R2 = Rparam * Rparam
    invR2 = 1.0 / R2

    # Create a container of PseudoJet objects
    history, Qtot = initial_history(initial_particles)
    jets = deepcopy(initial_particles)
    for ijet, jet in enumerate(jets):
        jet.info = BasicJetInfo(id = ijet, nn_dist=R2)

    # Create the numpy arrays corresponding to the pseudojets that will be used
    # for fast calculations
    npjets = NPPseudoJets(len(jets) * 2)
    npjets.set_jets(jets)

    # Setup the nearest neighbours, which is an expensive
    # initial operation (N^2 scaling here)
    scan_for_all_nearest_neighbours(npjets.phi, npjets.rap, npjets.inv_pt2, 
                                    npjets.dist, npjets.akt_dist, npjets.nn, npjets.mask, R2)

    # Each iteration we either merge two jets to one, or we
    # finalise a jet. Thus it takes a number of iterations
    # equal to the number of jets to finish
    for iteration in range(len(initial_particles)):
        distance, ijetA = find_closest_jets(npjets.akt_dist, npjets.nn)
        ijetB = npjets.nn[ijetA]
        # Add normalisation for real distance
        distance *= invR2

        if (ijetB >= 0):
            if ijetB < ijetA:
                ijetA, ijetB = ijetB, ijetA
            logger.debug(f"Iteration {iteration+1}: {distance} for jet {ijetA} and jet {ijetB}")

            # Merge jets
            npjets.mask_slot(ijetA)
            npjets.mask_slot(ijetB)
            merged_jet = jets[ijetA] + jets[ijetB]
            merged_jet.info = BasicJetInfo(id = len(jets), nn_dist=R2)
            jets.append(merged_jet)
            inewjet = npjets.insert_jet(merged_jet)
            add_step_to_history(history=history, jets=jets, 
                                parent1=jets[ijetA].cluster_hist_index,
                                parent2=jets[ijetB].cluster_hist_index,
                                jetp_index=merged_jet.info.id, distance=distance)
            
            # Get the NNs for the merged pseudojet
            scan_for_my_nearest_neighbours(inewjet, npjets.phi, npjets.rap, npjets.inv_pt2, 
                                           npjets.dist, npjets.akt_dist, npjets.nn, npjets.mask, R2)
        else:
            logger.debug(f"Iteration {iteration+1}: {distance} for jet {ijetA} and jet {ijetB}")
            # Beamjet
            npjets.mask_slot(ijetA)
            add_step_to_history(history=history, jets=jets, parent1=jets[ijetA].cluster_hist_index, 
                                parent2=BeamJet, 
                                jetp_index=Invalid, distance=distance)

        # Now need to update nearest distances, when pseudojets are unmasked and
        # had either jetA or jetB as their nearest neighbour
        if ijetB != -1:
            jets_to_update = np.logical_and(~npjets.mask, np.logical_or(npjets.nn == ijetA , npjets.nn == ijetB))
        else:
            jets_to_update = np.logical_and(~npjets.mask, npjets.nn == ijetA)
        ijets_to_update = np.where(jets_to_update)

        # Doable without actually needing a loop?
        for ijet_to_update in ijets_to_update[0]:
            scan_for_my_nearest_neighbours(ijet_to_update, npjets.phi, npjets.rap, npjets.inv_pt2, 
                                npjets.dist, npjets.akt_dist, npjets.nn, npjets.mask, R2)

        # Useful to check that we have done all updates correctly (only for debug!)
        if logger.level == logging.DEBUG:
            npjets_copy = deepcopy(npjets)
            scan_for_all_nearest_neighbours(npjets_copy.phi, npjets_copy.rap, npjets_copy.inv_pt2, 
                                npjets_copy.dist, npjets.akt_dist, npjets_copy.nn, npjets_copy.mask, R2)
            compare_status(npjets, npjets_copy)

    return inclusive_jets(jets, history, ptmin=ptmin)
