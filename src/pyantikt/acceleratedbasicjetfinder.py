from pyantikt.nphistory import NPHistory
from pyantikt.pseudojet import PseudoJet
from pyantikt.nppseudojet import NPPseudoJets

import logging
import numpy as np
import numpy.typing as npt

logger = logging.getLogger("jetfinder")

from numba import njit
from copy import deepcopy

Invalid = -3
NonexistentParent = -2
BeamJet = -1


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


def compare_status(working:NPPseudoJets, test:NPPseudoJets):
    '''Test two different copies of numpy pseudojet containers that should be equal'''
    dist_diff = working.akt_dist!=test.akt_dist
    idist_diff = np.where(dist_diff)
    if len(idist_diff[0]) > 0:
        print(f"Differences found after full scan of NNs: {idist_diff[0]}")
        for ijet in idist_diff[0]:
            print(f"{ijet}\nW: {working.print_jet(ijet)}\nT: {test.print_jet(ijet)}")
        raise RuntimeError("Jet sets are not the same and they should be!")


def add_step_to_history(history: NPHistory, jets: list[PseudoJet], 
                        parent1: int, parent2: int, jetp_index: int, distance: float):
    '''Add a merging step to the history of clustering
        history - list of HistoryElement entities
        jets - list of pseudojets
        parent1 - the *history* element which is the parent of this merger
        parent2 - the *history* element which is the parent of this merger (can be Invalid)
        jetp_index - the new pseudojet that results from this merger (if both parents exist)
        distance - the distance metric for this merge step
    '''
    max_dij_so_far = max(distance, history.max_dij_so_far[history.size-1])

    history.append(parent1=parent1, parent2=parent2, jetp_index=jetp_index, dij=distance,
                                    max_dij_so_far=max_dij_so_far)

    local_step = history.next-1
    logger.debug(f"Added history step {local_step}: {history.parent1[local_step]}")

    if parent1 >= 0:
        if history.child[parent1] != -1:
            raise (
                RuntimeError(
                    f"Internal error. Trying to recombine a parent1 object that has previsously been recombined: {parent1}"
                )
            )
    history.child[parent1] = local_step

    if parent2 >= 0:
        if history.child[parent2] != -1:
            raise (
                RuntimeError(
                    f"Internal error. Trying to recombine a parent1 object that has previsously been recombined: {parent2}"
                )
            )
    history.child[parent2] = local_step

    # get cross-referencing right from PseudoJets
    if jetp_index >= 0:
        jets[jetp_index].cluster_history_index = local_step

def inclusive_jets(jets: list[PseudoJet], history: NPHistory, ptmin:float=0.0):
    '''return all inclusive jets of a ClusterSequence with pt > ptmin'''
    dcut = ptmin * ptmin
    jets_local = list()
    # For inclusive jets with a plugin algorithm, we make no
    # assumptions about anything (relation of dij to momenta,
    # ordering of the dij, etc.)
    for elt in range(history.size-1, -1, -1):
        if history.parent2[elt] != BeamJet:
            continue
        iparent_jet = history.jetp_index[history.parent1[elt]]
        jet = jets[iparent_jet]
        if jet.pt2 >= dcut:
            jets_local.append(jet)

    return jets_local

def basicjetfinder(initial_particles: list[PseudoJet], Rparam: float=0.4, ptmin: float=0.0):
    """Basic AntiKt Jet finding code"""
    R2 = Rparam * Rparam
    invR2 = 1.0 / R2

    # Create a container of PseudoJet objects
    history = NPHistory(2 * len(initial_particles))
    Qtot = history.fill_initial_history(initial_particles)

    # Was doing a deepcopy here, but that turns out to be
    # 1. unnecessary
    # 2. extremely expensive
    jets = initial_particles

    # Create the numpy arrays corresponding to the pseudojets that will be used
    # for fast calculations
    npjets = NPPseudoJets(len(jets))
    npjets.set_jets(jets)

    # Setup the nearest neighbours, which is an expensive
    # initial operation (N^2 scaling here)
    scan_for_all_nearest_neighbours(npjets.phi, npjets.rap, npjets.inv_pt2, 
                                    npjets.dist, npjets.akt_dist, npjets.nn, 
                                    npjets.mask, R2)

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

            jet_indexA = npjets.jets_index[ijetA]
            jet_indexB = npjets.jets_index[ijetB]

            merged_jet = jets[jet_indexA] + jets[jet_indexB]
            imerged_jet = len(jets)
            jets.append(merged_jet)

            # We recycle the slot of jetA (which is the lowest slot)
            npjets.insert_jet(merged_jet, slot=ijetA, jet_index=imerged_jet)
            add_step_to_history(history=history, jets=jets, 
                                parent1=jets[jet_indexA].cluster_history_index,
                                parent2=jets[jet_indexB].cluster_history_index,
                                jetp_index=imerged_jet, distance=distance)
            
            # Get the NNs for the merged pseudojet
            scan_for_my_nearest_neighbours(ijetA, npjets.phi, npjets.rap, npjets.inv_pt2, 
                                           npjets.dist, npjets.akt_dist, npjets.nn, npjets.mask, R2)
        else:
            logger.debug(f"Iteration {iteration+1}: {distance} for jet {ijetA} and jet {ijetB}")
            # Beamjet
            npjets.mask_slot(ijetA)
            jet_indexA = npjets.jets_index[ijetA]
            add_step_to_history(history=history, jets=jets, parent1=jets[jet_indexA].cluster_history_index, 
                                parent2=BeamJet, 
                                jetp_index=Invalid, distance=distance)

        # Now need to update nearest distances, when pseudojets are unmasked and
        # had either jetA or jetB as their nearest neighbour
        # Note, it doesn't matter that we reused the ijetA slot here!
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
