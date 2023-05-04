from pyantikt.nphistory import NPHistory
from pyantikt.pseudojet import PseudoJet
from pyantikt.nptiling import TilingDef, NPTiling

import logging
import numpy as np
import numpy.typing as npt

logger = logging.getLogger("jetfinder")

from numba import njit
from sys import float_info
from math import trunc, floor

Invalid = -3
NonexistentParent = -2
BeamJet = -1


def determine_rapidity_extent(particles:list[PseudoJet]):
    """ "Have a binning of rapidity that goes from -nrap to nrap
    in bins of size 1; the left and right-most bins include
    include overflows from smaller/larger rapidities"""

    if len(particles) == 0:
        return 0.0, 0.0

    nrap = 20
    nbins = 2 * nrap
    counts = [0 for n in range(nbins)]

    # get the minimum and maximum rapidities and at the same time bin
    # the multiplicities as a function of rapidity to help decide how
    # far out it's worth going
    minrap = float_info.max
    maxrap = -float_info.max

    # As a timesaver, construct arrays of rapidity and phi here as we
    # are paying the price of the event loop anyway
    rap = np.empty(len(particles), dtype=float)
    phi = np.empty(len(particles), dtype=float)

    ibin = 0
    for ip, p in enumerate(particles):
        rap[ip] = p.rap
        phi[ip] = p.phi

        # ignore particles with infinite rapidity
        if p.E == abs(p.pz):
            continue

        y = p.rap
        minrap = min(minrap, y)
        maxrap = max(maxrap, y)

        # now bin the rapidity to decide how far to go with the tiling.
        # Remember the bins go from ibin=0 (rap=-infinity..-19)
        # to ibin = nbins-1 (rap=19..infinity for nrap=20)
        # This Python construct is a 'clamp' function (ensure that we pick a value from 0 to nbins-1)
        ibin = max(0, min(nbins - 1, trunc(y + nrap)))
        counts[ibin] += 1

    # now figure out the particle count in the busiest bin
    max_in_bin = max(counts)

    # and find minrap, maxrap such that edge bin never contains more
    # than some fraction of busiest, and at least a few particles; first do
    # it from left. NB: the thresholds chosen here are largely
    # guesstimates as to what might work.
    #
    # 2014-07-17: in some tests at high multiplicity (100k) and particles going up to
    #             about 7.3, anti-kt R=0.4, we found that 0.25 gave 20% better run times
    #             than the original value of 0.5.
    allowed_max_fraction = 0.25

    # the edge bins should also contain at least min_multiplicity particles
    min_multiplicity = 4

    # now calculate how much we can accumulate into an edge bin
    allowed_max_cumul = floor(max(max_in_bin * allowed_max_fraction, min_multiplicity))

    # make sure we don't require more particles in a bin than max_in_bin
    allowed_max_cumul = min(max_in_bin, allowed_max_cumul)

    # Start scan over rapidity bins from the left, to find out minimum rapidity of tiling
    # In this code, cumul2 isn't actually used anywhere
    cumul_lo = 0.0
    cumul2 = 0.0
    ibin_lo = 0
    while ibin_lo < nbins:
        cumul_lo += counts[ibin_lo]
        if cumul_lo >= allowed_max_cumul:
            minrap = max(minrap, ibin_lo - nrap)
            break
        ibin_lo += 1
    if ibin_lo == nbins:
        raise RuntimeError(
            "Failed to find a low bin"
        )  # internal consistency check that you found a bin
    cumul2 += cumul_lo**2

    # then do it from right, to find out maximum rapidity of tiling
    cumul_hi = 0.0
    ibin_hi = nbins - 1
    while ibin_hi >= 0:
        cumul_hi += counts[ibin_hi]
        if cumul_hi >= allowed_max_cumul:
            maxrap = min(maxrap, ibin_hi - nrap + 1)
            break
        ibin_hi -= 1
    if ibin_hi == -1:
        raise RuntimeError(
            "Failed to find a high bin"
        )  # internal consistency check that you found a bin

    # print(ibin_lo, cumul_lo, ibin_hi, cumul_hi)

    # consistency check
    if ibin_hi < ibin_lo:
        raise RuntimeError(
            "Low/high bins inconsistent"
        )  # internal consistency check that you found a bin

    # now work out cumul2
    if ibin_hi == ibin_lo:
        # if there is a single bin (potentially including overflows
        # from both sides), cumul2 is the square of the total contents
        # of that bin, which we obtain from cumul_lo and cumul_hi minus
        # the double counting of part that is contained in both
        # (putting double)
        cumul2 = (cumul_lo + cumul_hi - counts[ibin_hi]) ** 2
    else:
        # otherwise we have a straightforward sum of squares of bin
        # contents
        cumul2 += cumul_hi**2

    # now get the rest of the squared bin contents
    for ibin in range(ibin_lo + 1, ibin_hi + 1):
        cumul2 += counts[ibin] ** 2

    return minrap, maxrap, rap, phi


def initial_tiling(jets, Rparam=0.4):
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

    tiles_rap_min, tiles_rap_max, rap, phi = determine_rapidity_extent(jets)
    # print(tiles_rap_min, tiles_rap_max)

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
    myhist = np.histogram2d(rap, phi, bins=[n_tiles_rap, n_tiles_phi],
                            range=[[tiles_rap_min, tile_size_rap * n_tiles_rap + tiles_rap_min], [0.0, 2*np.pi]])
    # Can cross check with original here
    # print(myhist)

    max_jets_per_tile = np.int64(np.max(myhist[0]))
    # print(f"Max jets per tile: {max_jets_per_tile}")

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
    tiling.fill_with_jets(jets, rap, phi)

    return tiling

@njit
def tile_self_scan(irap:np.int64, iphi:np.int64,
                   rap:npt.ArrayLike, phi:npt.ArrayLike,
                   inv_pt2:npt.ArrayLike,
                   nn:npt.ArrayLike, dist:npt.ArrayLike,
                   akt_dist:npt.ArrayLike, 
                   mask:npt.ArrayLike, R2:npt.DTypeLike,
                   phidim:np.int64,
                   slotdim:np.int64):
    """Scan a tile for it's own nearest neighbours"""
    for islot in np.where(mask[irap,iphi]==False)[0]:
        # Saftey first...
        dist[irap,iphi,islot] = R2
        nn[irap,iphi,islot] = -1
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
            nn[irap,iphi,islot] = -1
            akt_dist[irap,iphi,islot] = dist[irap,iphi,islot] * inv_pt2[irap,iphi,islot]
        else:
            # This is a manual ravelling, as "ravel_multi_index" isn't supported in numba
            nn[irap,iphi,islot] = irap*(phidim*slotdim) + iphi*slotdim + iclosejet
            # nn[irap,iphi,islot] = np.ravel_multi_index((irap, iphi, iclosejet), nn.shape)
            akt_dist[irap,iphi,islot] = dist[irap,iphi,islot] * min(inv_pt2[irap,iphi,islot], inv_pt2[irap, iphi, iclosejet])


@njit
def tile_comparison_scan(irap:np.int64, iphi:np.int64,
                         jrap:np.int64, jphi:np.int64,
                         rap:npt.ArrayLike, phi:npt.ArrayLike,
                         nn:npt.ArrayLike, dist:npt.ArrayLike,
                         akt_dist:npt.ArrayLike, inv_pt2:npt.ArrayLike,
                         mask:npt.ArrayLike, R2:npt.DTypeLike,
                         phidim:np.int64,
                         slotdim:np.int64):
    """Scan a tile (irap, iphi) against jets in tile (jrap, jphi)"""
    for islot in np.where(mask[irap,iphi]==False)[0]:
        _dphi = np.pi - np.abs(np.pi - np.abs(phi[jrap,jphi] - phi[irap,iphi,islot]))
        _drap = rap[jrap,jphi] - rap[irap,iphi,islot]
        _dist = _dphi*_dphi + _drap*_drap
        _dist[mask[jrap,jphi]] = 1e20 # Don't consider any masked jets
        jclosejet = _dist.argmin()
        close_dist = _dist[jclosejet]
        if dist[irap,iphi,islot] > close_dist:
            dist[irap,iphi,islot] = close_dist
            # This is a manual ravelling, as "ravel_multi_index" isn't supported in numba
            nn[irap,iphi,islot] = jrap*(phidim*slotdim) + jphi*slotdim + jclosejet
            #nn[irap,iphi,islot] = np.ravel_multi_index((jrap,jphi,iclosejet), nn.shape)
            akt_dist[irap,iphi,islot] = dist[irap,iphi,islot] * min(inv_pt2[irap,iphi,islot], inv_pt2[jrap, jphi, jclosejet])


def scan_for_all_nearest_neighbours(nptiling:NPTiling, R2:npt.DTypeLike):
    """Scan over all tiles, where we need to self-scan, then compare against
    neighbours (as scanning is currently uni-directional, we cannot use the
    rightmost scanning strategy)"""
    for irap in range(nptiling.setup.n_tiles_rap):
        for iphi in range(nptiling.setup.n_tiles_phi):
            tile_self_scan(irap=irap, iphi=iphi, rap=nptiling.rap, phi=nptiling.phi, 
                            inv_pt2=nptiling.inv_pt2, nn=nptiling.nn, 
                            dist=nptiling.dist, akt_dist=nptiling.akt_dist,
                            mask=nptiling.mask, R2=R2,
                            phidim=nptiling.setup.n_tiles_phi,
                            slotdim=nptiling.max_jets_per_tile)

            # Now we scan all of the neighbour tiles
            for jrap, jphi in nptiling.neighbourtiles[irap,iphi]:
                if jrap == -1:
                    continue
                tile_comparison_scan(irap=irap, iphi=iphi, jrap=jrap, jphi=jphi,
                                     rap=nptiling.rap, phi=nptiling.phi, 
                            inv_pt2=nptiling.inv_pt2, nn=nptiling.nn, 
                            dist=nptiling.dist, akt_dist=nptiling.akt_dist,
                            mask=nptiling.mask, R2=R2,
                            phidim=nptiling.setup.n_tiles_phi,
                            slotdim=nptiling.max_jets_per_tile)

def find_closest_jets(akt_distance:npt.ArrayLike):
    minimum_distance_ravelled = np.argmin(akt_distance)
    minimum_distance_index = np.unravel_index(minimum_distance_ravelled, akt_distance.shape)
    return akt_distance[minimum_distance_index], tuple(minimum_distance_index), minimum_distance_ravelled


def scan_for_tile_nearest_neighbours(nptiling:NPTiling, newjetindex:tuple[int], R2:float):
    """Scan for nearest neighbours of a tile"""
    # First, scan my own tile
    irap = newjetindex[0]
    iphi = newjetindex[1]
    tile_self_scan(irap=irap, iphi=iphi,
                   rap=nptiling.rap, phi=nptiling.phi,
                   inv_pt2=nptiling.inv_pt2, nn=nptiling.nn,
                   dist=nptiling.dist,
                   akt_dist=nptiling.akt_dist, mask=nptiling.mask,
                   R2=R2,
                   phidim=nptiling.setup.n_tiles_phi,
                   slotdim=nptiling.max_jets_per_tile)

    # Now scan from my tile to surrounding tiles
    for jrap, jphi in nptiling.neighbourtiles[irap,iphi]:
        if jrap == -1:
            continue
        tile_comparison_scan(irap=irap, iphi=iphi, jrap=jrap, jphi=jphi,
                                rap=nptiling.rap, phi=nptiling.phi, 
                    inv_pt2=nptiling.inv_pt2, nn=nptiling.nn, 
                    dist=nptiling.dist, akt_dist=nptiling.akt_dist,
                    mask=nptiling.mask, R2=R2,
                    phidim=nptiling.setup.n_tiles_phi,
                    slotdim=nptiling.max_jets_per_tile)


def do_debug_scan(nptiling:NPTiling, ijet):
    print(f"Debug scan on {nptiling.dump_jet(ijet)}")
    print(f"NN Tiles: {nptiling.righttiles[ijet[0], ijet[1]]}")
    min_dist = 1e20
    nn = -1
    for irap in range(nptiling.setup.n_tiles_rap):
        for iphi in range(nptiling.setup.n_tiles_phi):
            for islot in np.where(nptiling.mask[irap,iphi]==False)[0]:
                _dphi = np.pi - np.abs(np.pi - np.abs(nptiling.phi[ijet] - nptiling.phi[irap,iphi,islot]))
                _drap = nptiling.rap[ijet] - nptiling.rap[irap,iphi,islot]
                _dist = _dphi*_dphi + _drap*_drap
                print(f"{irap},{iphi},{islot} -> {_dist}: ", end="")
                print(nptiling.dump_jet((irap,iphi,islot)))
                if min_dist > _dist and (irap != ijet[0] or iphi != ijet[1] or islot != ijet[2]):
                    min_dist = _dist
                    nn = np.ravel_multi_index((irap, iphi, islot), nptiling.nn.shape)
    print(f"Got minimum distance {min_dist} to {nn}")


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
    logger.debug(f"Added history step {local_step}: {parent1}, {parent2}, {distance}")

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

    # We scan the list of inital particles, setting up an appropriate
    # tiling structure and filling it with our initial particles
    nptiling = initial_tiling(jets, Rparam)

    # Setup the initial nearest neighbours
    scan_for_all_nearest_neighbours(nptiling, R2)

    # Each iteration we either merge two jets to one, or we
    # finalise a jet. Thus it takes a number of iterations
    # equal to the number of jets to finish
    for iteration in range(len(initial_particles)):
        # ijet{A,B = coordinate tuple for jet{A,B}
        # iflatjet{A,B} = flattened array index for jet {A,B}
        distance, ijetA, iflatjetA = find_closest_jets(nptiling.akt_dist)
        iflatjetB = nptiling.nn[ijetA]
        ijetB = np.unravel_index(iflatjetB, nptiling.nn.shape) if iflatjetB >= 0 else None

        # Add normalisation for real distance
        distance *= invR2

        # For each iteration, keep track of any recanned tiles, to avoid repetition
        tiles_rescanned = []

        if (iflatjetB >= 0):
            # This is the index in the PseudoJet array
            jet_indexA = nptiling.jets_index[ijetA]
            jet_indexB = nptiling.jets_index[ijetB]
            logger.debug(f"Iteration {iteration+1}: {distance} for jet {ijetA}={jet_indexA} and jet {ijetB}={jet_indexB}")
            if jet_indexB == -1:
                # Something very wrong
                raise RuntimeError(f"Invalid JetB state: {nptiling.dump_jet(ijetB)}")

            # Mask jets
            nptiling.mask_slot(ijetA)
            nptiling.mask_slot(ijetB)

            # Create merged jet
            merged_jet = jets[jet_indexA] + jets[jet_indexB]
            imerged_jet = len(jets)
            merged_jet.cluster_history_index = imerged_jet
            jets.append(merged_jet)

            # Insert the merged jet
            add_step_to_history(history=history, jets=jets, 
                                parent1=jets[jet_indexA].cluster_history_index,
                                parent2=jets[jet_indexB].cluster_history_index,
                                jetp_index=imerged_jet, distance=distance)

            newjetindex = nptiling.insert_jet(merged_jet, npjet_index=imerged_jet)

            # Get the NNs for the merged pseudojet
            # Note, this rescans the whole tile and all neighbours
            scan_for_tile_nearest_neighbours(nptiling, newjetindex, R2)
            tiles_rescanned.append((newjetindex[0], newjetindex[1]))
        else:
            jet_indexA = nptiling.jets_index[ijetA]
            logger.debug(f"Iteration {iteration+1}: {distance} for jet {ijetA}={jet_indexA} and beam")
            # Beamjet
            nptiling.mask_slot(ijetA)
            add_step_to_history(history=history, jets=jets, 
                                parent1=jets[jet_indexA].cluster_history_index, 
                                parent2=BeamJet, 
                                jetp_index=Invalid, distance=distance)
        


        # If there are any active jets which have jetA or jetB as their nearest
        # neighbour, then we need to rescan for them. Remember that we are scanning
        # tile-by-tile, so we also avoid useless rescans by keeping track of where
        # we have already rescanned        
        jets_to_update = np.where(nptiling.nn==iflatjetA)
        logger.debug(f"To update for jetA: {jets_to_update}")
        for irap, iphi, islot in zip(jets_to_update[0], jets_to_update[1], jets_to_update[2]):
            if (irap, iphi) not in tiles_rescanned:
                scan_for_tile_nearest_neighbours(nptiling, (irap, iphi, islot), R2)
                tiles_rescanned.append((irap, iphi))
        if (iflatjetB >= 0):
            jets_to_update = np.where(nptiling.nn==iflatjetB)
            logger.debug(f"To update for jetB: {jets_to_update}")
            for irap, iphi, islot in zip(jets_to_update[0], jets_to_update[1], jets_to_update[2]):
                if (irap, iphi) not in tiles_rescanned:
                    scan_for_tile_nearest_neighbours(nptiling, (irap, iphi, islot), R2)
                    tiles_rescanned.append((irap, iphi))

    return inclusive_jets(jets, history, ptmin=ptmin)
