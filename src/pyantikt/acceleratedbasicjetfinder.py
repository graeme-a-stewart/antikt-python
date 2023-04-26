
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

# Oh, jitting this function causes it to *fail*, not using masking properly!
#@njit
def find_closest_jets(dist: npt.ArrayLike, nn: npt.ArrayLike):
    '''Look over active jets and find the closest'''
    closest = dist.argmin()
    return dist[closest], closest


@njit
def scan_for_all_nearest_neighbours(phi: npt.ArrayLike, rap: npt.ArrayLike, inv_pt2: npt.ArrayLike,
                                    dist: npt.ArrayLike, nn: npt.ArrayLike, mask:npt.ArrayLike, R2:float):
    '''Do a full scan for nearest (geometrical) neighbours'''
    for ijet in range(phi.size):
        if bool(phi[ijet]) == False:
            continue
        _dphi = np.pi - np.abs(np.pi - (phi - phi[ijet]))
        _drap = rap - rap[ijet]
        _dist = _dphi*_dphi + _drap*_drap
        _dist[ijet] = R2
        iclosejet = _dist[~mask].argmin()
        dist[ijet] = _dist[iclosejet]
        if iclosejet == ijet:
            nn[ijet] = -1
            dist[ijet] *= inv_pt2[ijet]
        else:
            nn[ijet] = iclosejet
            dist[ijet] *= inv_pt2[ijet] if inv_pt2[ijet] < inv_pt2[iclosejet] else inv_pt2[iclosejet]

@njit
def scan_for_my_nearest_neighbours(ijet:int, phi: npt.ArrayLike, 
                                   rap:npt.ArrayLike, inv_pt2:npt.ArrayLike, 
                                   dist:npt.ArrayLike, nn:npt.ArrayLike, 
                                   mask:npt.ArrayLike, R2: float):
    '''Retest all other jets against the target jet'''
    _dphi = np.pi - np.abs(np.pi - (phi - phi[ijet]))
    _drap = rap - rap[ijet]
    _dist = _dphi*_dphi + _drap*_drap
    _dist[ijet] = R2
    iclosejet = _dist[~mask].argmin()
    dist[ijet] = _dist[iclosejet]
    if iclosejet == ijet:
        nn[ijet] = -1
        dist[ijet] *= inv_pt2[ijet]
    else:
        nn[ijet] = iclosejet
        dist[ijet] *= inv_pt2[ijet] if inv_pt2[ijet] < inv_pt2[iclosejet] else inv_pt2[iclosejet]
 

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
    logger.debug(f"Added history step {local_step}: {history[-1]}")

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
    logger.debug(npjets)

    # Setup the nearest neighbours, which is an expensive
    # initial operation (N^2 scaling here)
    scan_for_all_nearest_neighbours(npjets.phi, npjets.rap, npjets.inv_pt2, 
                                    npjets.dist, npjets.nn, npjets.dist.mask, R2)
    logger.debug(npjets)

    # Each iteration we either merge two jets to one, or we
    # finalise a jet. Thus it takes a number of iterations
    # equal to the number of jets to finish
    for iteration in range(len(initial_particles)):
        distance, ijetA = find_closest_jets(npjets.dist, npjets.nn)
        ijetB = npjets.nn[ijetA]
        # Add normalisation for real distance
        distance *= invR2
        logger.debug(f"Iteration {iteration+1}: {distance} for jet {ijetA} and jet {ijetB}")

        if (ijetB != -1):
            if ijetB < ijetA:
                ijetA, ijetB = ijetB, ijetA

            # Merge jets
            npjets.mask_slot(ijetA)
            npjets.mask_slot(ijetB)
            merged_jet = jets[ijetA] + jets[ijetB]
            merged_jet.info = BasicJetInfo(id = len(jets), nn_dist=R2)
            jets.append(merged_jet)
            inewjet = npjets.insert_jet(merged_jet)
            logger.debug(npjets)
            add_step_to_history(history=history, jets=jets, 
                                parent1=jets[ijetA].cluster_hist_index,
                                parent2=jets[ijetB].cluster_hist_index,
                                jetp_index=merged_jet.info.id, distance=distance)
            
            # Get the NNs for the merged pseudojet
            scan_for_my_nearest_neighbours(inewjet, npjets.phi, npjets.rap, npjets.inv_pt2, 
                                           npjets.dist, npjets.nn, npjets.dist.mask, R2)
            logger.debug(npjets)
        else:
            # Beamjet
            npjets.mask_slot(ijetA)
            add_step_to_history(history=history, jets=jets, parent1=jetA.cluster_hist_index, parent2=BeamJet, 
                                jetp_index=Invalid, distance=distance)

        # Now need to update nearest distances
        # for ijet, nn in np.ndenumerate(npjets.nn):
        #     print(ijet, nn)
        exit(0)

        # for jet in jets:
        #     if jet.info.active:
        #         # If a jet had jetA or JetB as NN, this is invalid now - rescan the active jets
        #         # If a jet had None as NN, test against the new jet
        #         # If a jet had an NN, but not A or B, test against the new jet
        #         if jet.info.nn == jetA.info.id:
        #             scan_for_my_nearest_neighbours(jet, jets, R2)
        #         elif jetB and jet.info.nn == jetB.info.id:
        #             scan_for_my_nearest_neighbours(jet, jets, R2)
        #         elif jetB:
        #             test_for_nearest_neighbour(jet, jets[-1])

    for jet in jets:
        if jet.info.active:
            print(f"Jet {jet.info.id} is still active")

    return inclusive_jets(jets, history, ptmin=ptmin)
