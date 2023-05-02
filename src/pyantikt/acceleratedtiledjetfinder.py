from pyantikt.nphistory import NPHistory
from pyantikt.pseudojet import PseudoJet
from pyantikt.nppseudojet import NPPseudoJets
from pyantikt.nptiling import TilingDef, NPTiling

import logging
import numpy as np
import numpy.typing as npt

logger = logging.getLogger("jetfinder")

from numba import njit
from copy import deepcopy
from sys import float_info

Invalid = -3
NonexistentParent = -2
BeamJet = -1

@njit
def determine_rapidity_extent(minrap:npt.DTypeLike, maxrap:npt.DTypeLike, rap:npt.ArrayLike):
    """Find the rapidity bining that gives a good distribution of initial
    particles. Bins are always size 1, and the left and right hand bins (which
    are "overflow" bins) also have ~0.25 of the maximum number of particles
    in any bin."""

    # Use the minimum and maximum rapidities to detemine the edge bins for counting
    # initial bin populations (we care about the RHS of the low bin, LHS of the high bin)
    min_rap_rhs = np.ceil(np.full(1, minrap))[0]
    max_rap_lhs = np.ceil(np.full(1, maxrap))[0]
    ibins = np.int64(max_rap_lhs - min_rap_rhs + 2)

    # print(minrap, maxrap, min_rap_rhs, max_rap_lhs)

    # now bin the rapidity to decide how far to go with the tiling.
    # Remember the bins go from ibin=0 (rap=-infinity..-floor(minrap))
    # to ibins-1 (rap=floor(maxrap)..infinity)
    bins = np.empty(ibins, dtype=float)
    bins[1:-1] = np.arange(min_rap_rhs, max_rap_lhs)
    bins[0] = np.finfo(np.float64).min
    bins[-1] = np.finfo(np.float64).max
    counts = np.histogram(rap, bins)[0]
    # print(counts)

    # now figure out the particle count in the busiest bin
    max_in_bin = np.max(counts)

    # the edge bins should also contain at least min_multiplicity particles
    min_multiplicity = np.int64(4)

    # and find minrap, maxrap such that edge bin never contains more
    # than some fraction of busiest, and at least a few particles; first do
    # it from left. NB: the thresholds chosen here are largely
    # guesstimates as to what might work.
    #
    # 2014-07-17: in some tests at high multiplicity (100k) and particles going up to
    #             about 7.3, anti-kt R=0.4, we found that 0.25 gave 20% better run times
    #             than the original value of 0.5.
    allowed_max_fraction = np.float64(0.25)
    allowed_max_cumul = allowed_max_fraction * max_in_bin
    if allowed_max_cumul < min_multiplicity:
        allowed_max_cumul = min_multiplicity

    min_bin = 0
    cummulated_min_bin = counts[0]
    while True:
        min_bin += 1
        cummulated_min_bin += counts[min_bin]
        if cummulated_min_bin >= allowed_max_cumul:
            break
    setminrap = np.int64(min_rap_rhs + min_bin - 1)

    max_bin = -1
    cummulated_max_bin = counts[-1]
    while True:
        max_bin -= 1
        cummulated_max_bin += counts[max_bin]
        if cummulated_max_bin >= allowed_max_cumul:
            break
    setmaxrap = np.int64(max_rap_lhs + max_bin + 1)

    # print(min_bin, cummulated_min_bin, max_bin, cummulated_max_bin)
    # print(setminrap, setmaxrap)

    return setminrap, setmaxrap

def initial_tiling(npjets, Rparam=0.4):
    """Decide on a tiling strategy"""

    # first decide tile sizes (with a lower bound to avoid huge memory use with
    # very small R)
    if 0.1 > Rparam:
        tile_size_rap = np.float64(0.1)
    else:
        tile_size_rap = np.float64(Rparam)
    # print(tile_size_eta, type(tile_size_eta))

    # it makes no sense to go below 3 tiles in phi -- 3 tiles is
    # sufficient to make sure all pair-wise combinations up to pi in
    # phi are possible
    n_tiles_phi = np.int64(np.floor(2.0 * np.pi / tile_size_rap))
    if n_tiles_phi < 3:
        n_tiles_phi = np.int64(3)

    tile_size_phi = np.float64(2.0) * np.pi / n_tiles_phi  # >= Rparam and fits in 2pi

    # Calling this function with broken out numpy types allows it to be jitted
    tiles_rap_min, tiles_rap_max = determine_rapidity_extent(np.min(npjets.rap), np.max(npjets.rap), npjets.rap)
    print(tiles_rap_min, tiles_rap_max)

    # now adjust the values
    tiles_irap_min = np.int64(np.floor(tiles_rap_min / tile_size_rap))
    tiles_irap_max = np.int64(np.floor(tiles_rap_max / tile_size_rap))
    tiles_rap_min = tiles_irap_min * tile_size_rap
    tiles_rap_max = tiles_irap_max * tile_size_rap
    n_tiles_rap = np.int64(tiles_irap_max - tiles_irap_min + 1)

    # print(tiles_rap_min,
    #     tiles_rap_max,
    #     tile_size_rap,
    #     tile_size_phi,
    #     n_tiles_rap,
    #     n_tiles_phi,
    #     tiles_irap_min,
    #     tiles_irap_max,
    # )

    # We need to do a quick scan, using the tiled definition to find out
    # how many jets we need to have space for in the tile, i.e., what's the
    # maximum value in any tile - we basically 2D historgam all the jets
    # and see what the maximum bin value is
    myhist = np.histogram2d(npjets.rap, npjets.phi, bins=[n_tiles_rap, n_tiles_phi],
                            range=[[tiles_rap_min, tile_size_rap * n_tiles_rap + tiles_rap_min], [0.0, 2*np.pi]])
    # Can cross check with original here
    # print(myhist)

    max_jets_per_tile = np.int64(np.max(myhist[0]))
    print(f"Max jets per tile: {max_jets_per_tile}")

    tiling_setup = TilingDef(
        tiles_rap_min,
        tiles_rap_max,
        tile_size_rap,
        tile_size_phi,
        n_tiles_rap,
        n_tiles_phi,
        tiles_irap_min,
        tiles_irap_max,
    )

    # allocate the tiles
    tiling = NPTiling(tiling_setup, max_jets_per_tile)
    tiling.fill_with_jets(npjets)

    return tiling

@njit
def find_closest_jets(akt_dist:npt.ArrayLike, nn:npt.ArrayLike):
    '''Look over active jets and find the closest'''
    closest = akt_dist.argmin()
    return akt_dist[closest], closest

@njit
def tile_self_scan(irap:np.int64, iphi:np.int64,
                   rap:npt.ArrayLike, phi:npt.ArrayLike,
                   inv_pt2:npt.ArrayLike,
                   nn:npt.ArrayLike, dist:npt.ArrayLike,
                   akt_dist:npt.ArrayLike, 
                   mask:npt.ArrayLike, R2:npt.DTypeLike):
    """Scan a tile for it's own nearest neighbours"""
    for islot in np.where(mask[irap,iphi]==False)[0]:
        # print(islot, type(islot))
        # _myphi = phi[irap,iphi,islot]
        # print(_myphi, type(_myphi))
        _dphi = np.pi - np.abs(np.pi - np.abs(phi[irap,iphi] - phi[irap,iphi,islot]))
        _drap = rap[irap,iphi] - np.float64(rap[irap,iphi,islot])
        _dist = _dphi*_dphi + _drap*_drap
        _dist[islot] = R2 # Avoid measuring the distance 0 to myself!
        _dist[mask[irap,iphi]] = 1e20 # Don't consider any masked jets
        iclosejet = _dist.argmin()
        dist[irap,iphi,islot] = _dist[iclosejet]
        if iclosejet == islot:
            nn[irap,iphi,islot] = (-1,-1,-1)
            akt_dist[irap,iphi,islot] = dist[irap,iphi,islot] * inv_pt2[irap,iphi,islot]
        else:
            nn[irap,iphi,islot] = (irap, iphi, iclosejet)
            akt_dist[irap,iphi,islot] = dist[irap,iphi,islot] * min(inv_pt2[irap,iphi,islot], inv_pt2[irap, iphi, iclosejet])
            if dist[irap,iphi,iclosejet] > dist[irap,iphi,islot]:
                nn[irap,iphi,iclosejet] = (irap,iphi,islot)
                dist[irap,iphi,iclosejet] = dist[irap,iphi,islot]
                akt_dist[irap,iphi,islot] = akt_dist[irap,iphi,islot]

@njit
def tile_comparison_scan(irap:np.int64, iphi:np.int64,
                         jrap:np.int64, jphi:np.int64,
                         rap:npt.ArrayLike, phi:npt.ArrayLike,
                         nn:npt.ArrayLike, dist:npt.ArrayLike,
                         akt_dist:npt.ArrayLike, inv_pt2:npt.ArrayLike,
                         mask:npt.ArrayLike, R2:npt.DTypeLike):
    """Scan a tile (irap, iphi) against jets in tile (jrap, jphi)"""
    for islot in np.where(mask[irap,iphi]==False)[0]:
        _dphi = np.pi - np.abs(np.pi - np.abs(phi[jrap,jphi] - phi[irap,iphi,islot]))
        _drap = rap[jrap,jphi] - rap[irap,iphi,islot]
        _dist = _dphi*_dphi + _drap*_drap
        _dist[mask[jrap,jphi]] = 1e20 # Don't consider any masked jets
        iclosejet = _dist.argmin()
        close_dist = _dist[iclosejet]
        if dist[irap,iphi,islot] > close_dist:
            dist[irap,iphi,islot] = close_dist
            nn[irap,iphi,islot] = (jrap,jphi,iclosejet)
            akt_dist[irap,iphi,islot] = dist[irap,iphi,islot] * min(inv_pt2[irap,iphi,islot], inv_pt2[jrap, jphi, iclosejet])
        if dist[jrap,jphi,iclosejet] > close_dist:
            dist[jrap,jphi,iclosejet] = close_dist
            nn[jrap,jphi,iclosejet] = (irap,iphi,islot)
            akt_dist[jrap,jphi,iclosejet] = akt_dist[irap,iphi,islot]

def scan_for_all_nearest_neighbours(nptiling:NPTiling, R2:npt.DTypeLike):
    """Scan over all tiles, working down and to the right, making
    all pair combinations of jets"""
    for irap in range(nptiling.setup.n_tiles_rap):
        for iphi in range(nptiling.setup.n_tiles_phi):
            tile_self_scan(irap=irap, iphi=iphi, rap=nptiling.rap, phi=nptiling.phi, 
                            inv_pt2=nptiling.inv_pt2, nn=nptiling.nn, 
                            dist=nptiling.dist, akt_dist=nptiling.akt_dist,
                            mask=nptiling.mask, R2=R2)

            # Now we scan all of the rightmost neighbour tiles
            for jrap, jphi in nptiling.righttiles[irap,iphi]:
                if jrap == -1:
                    continue
                tile_comparison_scan(irap=irap, iphi=iphi, jrap=jrap, jphi=jphi,
                                     rap=nptiling.rap, phi=nptiling.phi, 
                            inv_pt2=nptiling.inv_pt2, nn=nptiling.nn, 
                            dist=nptiling.dist, akt_dist=nptiling.akt_dist,
                            mask=nptiling.mask, R2=R2)

def find_closest_jets(akt_distance:npt.ArrayLike):
    minimum_distance_index = np.unravel_index(np.argmin(akt_distance), akt_distance.shape)
    return akt_distance[minimum_distance_index], minimum_distance_index


def scan_for_newjet_nearest_neighbours(nptiling:NPTiling, newjetindex:tuple[int], R2:float):
    """Scan for nearest neighbours of a new merged jet"""
    # First, scan my own tile
    print(f"Doing self scan for {newjetindex}")
    tile_self_scan(irap=newjetindex[0], iphi=newjetindex[1],
                   rap=nptiling.rap, phi=nptiling.phi,
                   inv_pt2=nptiling.inv_pt2, nn=nptiling.nn,
                   dist=nptiling.dist,
                   akt_dist=nptiling.akt_dist, mask=nptiling.mask,
                   R2=R2)

    # Now scan neighboring tiles
    # N.B. this is all tiles, not just rightmost
    for dirap in range(-1,2):
        for diphi in range(-1,2):
            jrap = newjetindex[0] + dirap
            jphi = newjetindex[1] + diphi
            print(f"Scanning {jrap}, {jphi} ({dirap}, {diphi})")
            if jrap < 0 or jrap > nptiling.setup.n_tiles_rap-1:
                continue
            if dirap==0 and diphi==0:
                continue
            if jphi < 0:
                jphi = nptiling.setup.n_tiles_phi-1
            if jphi > nptiling.setup.n_tiles_phi-1:
                jphi =0
            print(f"Scanning {jrap}, {jphi}")
            tile_comparison_scan(irap=newjetindex[0], iphi=newjetindex[1], jrap=jrap, jphi=jphi,
                rap=nptiling.rap, phi=nptiling.phi,
                inv_pt2=nptiling.inv_pt2, nn=nptiling.nn,
                dist=nptiling.dist,
                akt_dist=nptiling.akt_dist, mask=nptiling.mask,
                R2=R2)

# @njit
# def scan_for_my_nearest_neighbours(ijet:int, phi:npt.ArrayLike, 
#                                    rap:npt.ArrayLike, inv_pt2:npt.ArrayLike, 
#                                    dist:npt.ArrayLike, akt_dist:npt.ArrayLike, nn:npt.ArrayLike, 
#                                    mask:npt.ArrayLike, R2: float):
#     '''Retest all other jets against the target jet'''
#     nn[ijet] = -1
#     dist[ijet] = R2
#     _dphi = np.pi - np.abs(np.pi - np.abs(phi - phi[ijet]))
#     _drap = rap - rap[ijet]
#     _dist = _dphi*_dphi + _drap*_drap
#     _dist[ijet] = R2 # Avoid measuring the distance 0 to myself!
#     _dist[mask] = 1e20 # Don't consider any masked jets
#     iclosejet = _dist.argmin()
#     dist[ijet] = _dist[iclosejet]
#     if iclosejet == ijet:
#         nn[ijet] = -1
#         akt_dist[ijet] = dist[ijet] * inv_pt2[ijet]
#     else:
#         nn[ijet] = iclosejet
#         akt_dist[ijet] = dist[ijet] * (inv_pt2[ijet] if inv_pt2[ijet] < inv_pt2[iclosejet] else inv_pt2[iclosejet])
#         # As this function is called on new PseudoJets it's possible
#         # that we are now the NN of our NN
#         if dist[iclosejet] > dist[ijet]:
#             dist[iclosejet] = dist[ijet]
#             nn[iclosejet] = ijet
#             akt_dist[iclosejet] = dist[iclosejet] * (inv_pt2[ijet] if inv_pt2[ijet] < inv_pt2[iclosejet] else inv_pt2[iclosejet])


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
        jets[jetp_index].cluster_hist_index = local_step

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

def faster_tiled_N2_cluster(initial_particles: list[PseudoJet], Rparam: float=0.4, ptmin: float=0.0):
    """Accelerated tiled AntiKt Jet finding code"""
    R2 = Rparam * Rparam
    invR2 = 1.0 / R2

    # Create a container of PseudoJet objects
    history = NPHistory(2 * len(initial_particles))
    Qtot = history.fill_initial_history(initial_particles)

    # Was originally doing a deepcopy here, but that turns out to be
    # 1. unnecessary
    # 2. extremely expensive
    jets = initial_particles

    # Create the numpy arrays corresponding to the pseudojets that will be used
    # for fast calculations
    npjets = NPPseudoJets(len(jets))
    npjets.set_jets(jets)

    # We scan the list of inital particles, setting up an appropriate
    # tiling structure and filling it with our initial particles
    nptiling = initial_tiling(npjets, Rparam)

    # Setup the initial nearest neighbours
    scan_for_all_nearest_neighbours(nptiling, R2)

    # Each iteration we either merge two jets to one, or we
    # finalise a jet. Thus it takes a number of iterations
    # equal to the number of jets to finish
    for iteration in range(len(initial_particles)):
        distance, ijetA = find_closest_jets(nptiling.akt_dist)
        ijetB = tuple(nptiling.nn[ijetA]) # Seems we need to force this to a tuple!
        # Add normalisation for real distance
        distance *= invR2

        print(f"{ijetA} and {ijetB} - distance {distance}")

        if (ijetB[0] >= 0):
            logger.debug(f"Iteration {iteration+1}: {distance} for jet {ijetA} and jet {ijetB}")
            jet_indexA = nptiling.jets_index[ijetA]
            jet_indexB = nptiling.jets_index[ijetB]

            logger.debug(f"Mapping to {jet_indexA} and {jet_indexB}")

            # Mask jets
            nptiling.mask_slot(ijetA)
            nptiling.mask_slot(ijetB)

            # Create merged jet
            merged_jet = jets[jet_indexA] + jets[jet_indexB]
            imerged_jet = len(jets)
            jets.append(merged_jet)

            # Insert the merged jet
            add_step_to_history(history=history, jets=jets, 
                                parent1=jets[jet_indexA].cluster_hist_index,
                                parent2=jets[jet_indexB].cluster_hist_index,
                                jetp_index=imerged_jet, distance=distance)

            newjetindex = nptiling.insert_jet(merged_jet, npjet_index=imerged_jet)

            # Get the NNs for the merged pseudojet
            # Note, this rescans the whole tile and all neighbours
            scan_for_newjet_nearest_neighbours(nptiling, newjetindex, R2)
            exit(0)
        else:
            logger.debug(f"Iteration {iteration+1}: {distance} for jet {ijetA} and jet (-1,-1,-1)")
            jet_indexA = nptiling.jets_index[ijetA]
            logger.debug(f"Mapping to {jet_indexA} and -1")
            # Beamjet
            nptiling.mask_slot(ijetA)
            jet_indexA = nptiling.jets_index[ijetA]
            add_step_to_history(history=history, jets=jets, parent1=jets[jet_indexA].cluster_hist_index, 
                                parent2=BeamJet, 
                                jetp_index=Invalid, distance=distance)

        # If there are any remiaining active jets which have ijetA or ijetB as their nearest
        # neighbour, then we need to rescan for them. Mostly this should be taken care of by
        # any new pseudojet scan in the case of a merger, but we need to check



        # Now need to update nearest distances, when pseudojets are unmasked and
        # had either jetA or jetB as their nearest neighbour
        # Note, it doesn't matter that we reused the ijetA slot here!
        # if ijetB != -1:
        #     jets_to_update = np.logical_and(~npjets.mask, np.logical_or(npjets.nn == ijetA , npjets.nn == ijetB))
        # else:
        #     jets_to_update = np.logical_and(~npjets.mask, npjets.nn == ijetA)
        # ijets_to_update = np.where(jets_to_update)

        # # Doable without actually needing a loop?
        # for ijet_to_update in ijets_to_update[0]:
        #     scan_for_my_nearest_neighbours(ijet_to_update, npjets.phi, npjets.rap, npjets.inv_pt2, 
        #                         npjets.dist, npjets.akt_dist, npjets.nn, npjets.mask, R2)

        # # Useful to check that we have done all updates correctly (only for debug!)
        # if logger.level == logging.DEBUG:
        #     npjets_copy = deepcopy(npjets)
        #     scan_for_all_nearest_neighbours(npjets_copy.phi, npjets_copy.rap, npjets_copy.inv_pt2, 
        #                         npjets_copy.dist, npjets.akt_dist, npjets_copy.nn, npjets_copy.mask, R2)
        #     compare_status(npjets, npjets_copy)

    return inclusive_jets(jets, history, ptmin=ptmin)
