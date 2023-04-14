
from copy import deepcopy
from dataclasses import dataclass
from math import pi
from pyantikt.history import HistoryElement, ClusterSequence, initial_history, HistoryState
from pyantikt.pseudojet import PseudoJet
from sys import float_info

import logging

logger = logging.getLogger(__name__)

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

def find_closest_jets(jets: list[PseudoJet]):
    '''Scan over active jets and find the closest'''
    closest = -1
    akt_dist_closest = float_info.max
    for jet in jets:
        if not jet.info.active:
            continue
        if jet.info.akt_dist < akt_dist_closest:
            closest = jet.info.id
            akt_dist_closest = jet.info.akt_dist

    return akt_dist_closest, jets[closest]


def scan_for_all_nearest_neighbours(jets: list[PseudoJet]):
    '''Do a full scan for nearest (geometrical) neighbours'''
    for ijetA, jetA in enumerate(jets):
        for ijetB, jetB in enumerate(jets[ijetA+1:], start=ijetA+1):
            dist = geometric_distance(jetA.phi, jetA.rap, jetB.phi, jetB.rap)
            if dist < jetA.info.nn_dist:
                jetA.info.nn_dist = dist
                jetA.info.nn = ijetB
            if dist < jetB.info.nn_dist:
                jetB.info.nn_dist = dist
                jetB.info.nn = ijetA
        jetA.info.akt_dist = antikt_distance(jetA.info.nn_dist, jetA.inv_pt2, jets[jetA.info.nn].inv_pt2 if jetA.info.nn else -1.0)


def scan_for_my_nearest_neighbours(jetA: PseudoJet, jets: list[PseudoJet], R2: float):
    '''Retest all other jets against the target jet'''
    jetA.info.nn = None
    jetA.info.nn_dist = R2
    for ijetB, jetB in enumerate(jets):
        if not jetB.info.active:
            continue
        if ijetB == jetA.info.id:
            continue
        dist = geometric_distance(jetA.phi, jetA.rap, jetB.phi, jetB.rap)
        if dist < jetA.info.nn_dist:
            jetA.info.nn_dist = dist
            jetA.info.nn = ijetB
    jetA.info.akt_dist = antikt_distance(jetA.info.nn_dist, jetA.inv_pt2, jets[jetA.info.nn].inv_pt2 if jetA.info.nn else -1.0)


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

    # Setup the nearest neighbours, which is an expensive
    # initial operation (N^2 scaling here)
    scan_for_all_nearest_neighbours(jets)

    # Each iteration we either merge two jets to one, or we
    # finalise a jet. Thus it takes a number of iterations
    # equal to the number of jets to finish
    for iteration in range(len(initial_particles)):
        distance, jetA = find_closest_jets(jets)

        # Add normalisation for real distance
        distance *= invR2
        logger.debug(f"Iteration {iteration+1}: {distance} for jet {jetA.info.id} and jet {jetA.info.nn}")
        jetB = jets[jetA.info.nn] if (jetA.info.nn != None) else None

        if (jetB):
            if jetB.info.id < jetA.info.id:
                jetA, jetB = jetB, jetA

            # Merge jets
            jetA.info.active = jetB.info.active = False
            merged_jet = jetA + jetB
            merged_jet.info = BasicJetInfo(id = len(jets), nn_dist=R2)
            jets.append(merged_jet)

            add_step_to_history(history=history, jets=jets, 
                                parent1=jetA.cluster_hist_index,
                                parent2=jetB.cluster_hist_index,
                                jetp_index=merged_jet.info.id, distance=distance)
            
            # Get the NNs for the merged pseudojet
            scan_for_my_nearest_neighbours(jets[-1], jets, R2)
        
        else:
            # Beamjet
            jetA.info.active = False
            add_step_to_history(history=history, jets=jets, parent1=jetA.cluster_hist_index, parent2=BeamJet, 
                                jetp_index=Invalid, distance=distance)

        # Now need to update nearest distances
        for jet in jets:
            if jet.info.active:
                # If a jet had jetA or JetB as NN, this is invalid now - rescan the active jets
                # If a jet had None as NN, test against the new jet
                # If a jet had an NN, but not A or B, test against the new jet
                if jet.info.nn == jetA.info.id:
                    scan_for_my_nearest_neighbours(jet, jets, R2)
                elif jetB and jet.info.nn == jetB.info.id:
                    scan_for_my_nearest_neighbours(jet, jets, R2)
                elif jetB:
                    test_for_nearest_neighbour(jet, jets[-1])

    for jet in jets:
        if jet.info.active:
            print(f"Jet {jet.info.id} is still active")

    return inclusive_jets(jets, history, ptmin=ptmin)
