
from copy import deepcopy
from dataclasses import dataclass
from math import pi
from pyantikt.history import HistoryElement, ClusterSequence, initial_history
from pyantikt.pseudojet import PseudoJet
from sys import float_info

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

def geometric_distance(jetA: PseudoJet, jetB: PseudoJet):
    """Distance between two jets"""
    dphi = pi - abs(pi - abs(jetA.phi - jetB.phi))
    drap = jetA.rap - jetB.rap
    return dphi * dphi + drap * drap

def antikt_distance(jetA: PseudoJet, jetB: PseudoJet | None):
    '''AntiKt distance between two jets'''
    if jetB:
        inv_pt2 = min(jetA.inv_pt2, jetB.inv_pt2)
    else:
        inv_pt2 = jetA.inv_pt2
    return jetA.info.nn_dist * inv_pt2

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
            dist = geometric_distance(jetA, jetB)
            if dist < jetA.info.nn_dist:
                jetA.info.nn_dist = dist
                jetA.info.nn = ijetB
            if dist < jetB.info.nn_dist:
                jetB.info.nn_dist = dist
                jetB.info.nn = ijetA
        jetA.info.akt_dist = antikt_distance(jetA, jets[jetA.info.nn] if jetA.info.nn else None)


def scan_for_my_nearest_neighbours(jetA: PseudoJet, jets: list[PseudoJet]):
    '''Retest all other jets against the target jet'''
    for ijetB, jetB in enumerate(jets):
        if not jetB.info.active:
            continue
        if ijetB == jetA.info.id:
            continue
        dist = geometric_distance(jetA, jetB)
        if dist < jetA.info.nn_dist:
            jetA.info.nn_dist = dist
            jetA.info.nn = ijetB
    jetA.info.akt_dist = antikt_distance(jetA, jets[jetA.info.nn] if jetA.info.nn else None)


def test_for_nearest_neighbour(jetA: PseudoJet, jetB: PseudoJet):
    '''Test two jets and see if they are nearest neighbours'''
    if jetA.info.id == jetB.info.id:
        return
    dist = geometric_distance(jetA, jetB)
    if dist < jetA.info.nn_dist:
        jetA.info.nn_dist = dist
        jetA.info.nn = jetB.info.nn
        jetA.info.akt_dist = antikt_distance(jetA, jetB)
    if dist < jetB.info.nn_dist:
        jetB.info.nn_dist = dist
        jetB.info.nn = jetA.info.id
        jetB.info.akt_dist = antikt_distance(jetB, jetA)


def inclusive_jets(jets: list[PseudoJet], history: list[HistoryElement], ptmin:float=0.0):
    """return all inclusive jets of a ClusterSequence with pt > ptmin"""
    dcut = ptmin * ptmin
    jets_local = list()
    # For inclusive jets with a plugin algorithm, we make no
    # assumptions about anything (relation of dij to momenta,
    # ordering of the dij, etc.)
    for elt in reversed(history):
        if elt.parent2 != -1:
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
    # print(history, Qtot)

    cs = ClusterSequence(jets, history, None, Qtot)

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
        # print(f"Iteration {iteration}: {distance} for jet {jetA.info.id} and jet {jetA.info.nn}")
        jetB = jets[jetA.info.nn] if jetA.info.nn else None

        if (jetB):
            # Merge jets
            jetA.info.active = jetB.info.active = False
            merged_jet = jetA + jetB
            merged_jet.info = BasicJetInfo(id = len(jets), nn_dist=R2)
            jets.append(merged_jet)
            scan_for_my_nearest_neighbours(jets[-1], jets)

            history.append(HistoryElement(parent1=jetA.info.id, parent2=jetB.info.id,
                                          jetp_index=merged_jet.info.id, dij=distance,
                                          max_dij_so_far=distance if history[-1].max_dij_so_far < distance else history[-1].max_dij_so_far))
            # print(history[-1], jets[-1])
        
        else:
            # Beamjet
            jetA.info.active = False
            history.append(HistoryElement(parent1=jetA.info.id, parent2=BeamJet, jetp_index=-1, dij=distance,
                                          max_dij_so_far=distance if history[-1].max_dij_so_far < distance else history[-1].max_dij_so_far))
            # print(history[-1])

        # Now need to update nearest distances
        for jet in jets:
            if jet.info.active:
                # If a jet had A or B as an NN, this is now invalid - scan all remaining jets
                # If a jet had None as NN, test against the new jet
                # If a jet had an NN, but not A or B, test against the new jet
                if jet.info.nn == jetA.info.id:
                    jet.info.nn_dist = R2
                    jet.info.nn = None
                    scan_for_my_nearest_neighbours(jet, jets)
                elif jetB and jet.info.nn == jetB.info.id:
                    jet.info.nn_dist = R2
                    jet.info.nn = None
                    scan_for_my_nearest_neighbours(jet, jets)
                elif jetB:
                    test_for_nearest_neighbour(jet, jets[-1])

    for jet in jets:
        if jet.info.active:
            print(f"Jet {jet.info.id} is still active")

    return inclusive_jets(jets, history, ptmin=ptmin)
