from copy import deepcopy
from math import pi, floor, trunc
from sys import float_info, exit
from pyantikt.tiles import (
    TilingDef,
    TiledJet,
    Tiling,
    rightneighbours,
    add_untagged_neighbours_to_tile_union,
    surrounding_tiles,
)
from pyantikt.history import HistoryElement, ClusterSequence, initial_history

import logging
logger = logging.getLogger("jetfinder")

Invalid = -3
NonexistentParent = -2
BeamJet = -1


def tiledjet_dist(jetA: TiledJet, jetB: TiledJet):
    """Distance between two tiled jets"""
    dphi = pi - abs(pi - abs(jetA.phi - jetB.phi))
    deta = jetA.eta - jetB.eta
    return dphi * dphi + deta * deta


def tiledjet_diJ(jet):
    kt2 = jet.kt2
    if jet.NN and (jet.NN.kt2 < kt2):
        kt2 = jet.NN.kt2
    return jet.NN_dist * kt2


def determine_rapidity_extent(particles):
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

    ibin = 0
    for p in particles:
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

    return minrap, maxrap


def initial_tiling(particles, Rparam=0.4):
    """Decide on a tiling strategy"""

    # first decide tile sizes (with a lower bound to avoid huge memory use with
    # very small R)
    tile_size_eta = max(0.1, Rparam)

    # it makes no sense to go below 3 tiles in phi -- 3 tiles is
    # sufficient to make sure all pair-wise combinations up to pi in
    # phi are possible
    n_tiles_phi = max(3, floor(2.0 * pi / tile_size_eta))

    tile_size_phi = 2 * pi / n_tiles_phi  # >= Rparam and fits in 2pi

    tiles_eta_min, tiles_eta_max = determine_rapidity_extent(particles)

    # now adjust the values
    tiles_ieta_min = floor(tiles_eta_min / tile_size_eta)
    tiles_ieta_max = floor(
        tiles_eta_max / tile_size_eta
    )  # FIXME shouldn't it be ceil ?
    tiles_eta_min = tiles_ieta_min * tile_size_eta
    tiles_eta_max = tiles_ieta_max * tile_size_eta
    n_tiles_eta = tiles_ieta_max - tiles_ieta_min + 1

    # print(
    #     tiles_eta_min,
    #     tiles_eta_max,
    #     tile_size_eta,
    #     tile_size_phi,
    #     n_tiles_eta,
    #     n_tiles_phi,
    #     tiles_ieta_min,
    #     tiles_ieta_max,
    # )
    # exit(0)

    tiling_setup = TilingDef(
        tiles_eta_min,
        tiles_eta_max,
        tile_size_eta,
        tile_size_phi,
        n_tiles_eta,
        n_tiles_phi,
        tiles_ieta_min,
        tiles_ieta_max,
    )

    # allocate the tiles
    return Tiling(tiling_setup)


def tiledjet_remove_from_tiles(tiling, jet):
    """Remove a jet from a tiling"""
    tile_indices = jet.tile_index

    # Now we need to find where the jet is in the tiled list (which is not cached)
    tile_jet_position = -1
    for ijet, tile_jet in enumerate(tiling.tiles[tile_indices[0]][tile_indices[1]]):
        # previous and next are references to the actual tiled jets, None if there is nothing
        if jet.id == tile_jet.id:
            tile_jet_position = ijet
            break
    if tile_jet_position == -1:
        raise (RuntimeError(f"Failed to find jet {jet.id} in tile {tile_indices}"))

    # Now we remove this jet, but set the forward and backward links
    # (Probably not needed as we're not really using the linked list structure in Python)
    if len(tiling.tiles[tile_indices[0]][tile_indices[1]]) == 1:
        # I'm the only one, so set to empty list
        tiling.tiles[tile_indices[0]][tile_indices[1]] = list()
    else:
        if tile_jet_position == 0:
            # Head jet, set the next jet's previous to None
            tiling.tiles[tile_indices[0]][tile_indices[1]][1].previous = None
        elif (
            tile_jet_position == len(tiling.tiles[tile_indices[0]][tile_indices[1]]) - 1
        ):
            # Tail jet, set the previous jet's next to None
            tiling.tiles[tile_indices[0]][tile_indices[1]][-2].next = None
        else:
            # In the middle somewhere, link up our jets on either side to each other
            tiling.tiles[tile_indices[0]][tile_indices[1]][
                tile_jet_position - 1
            ].next = tiling.tiles[tile_indices[0]][tile_indices[1]][
                tile_jet_position + 1
            ]
            tiling.tiles[tile_indices[0]][tile_indices[1]][
                tile_jet_position + 1
            ].previous = tiling.tiles[tile_indices[0]][tile_indices[1]][
                tile_jet_position - 1
            ]
        # Tidy myself up and then do the removal
        jet.next = jet.previous = None
        del tiling.tiles[tile_indices[0]][tile_indices[1]][tile_jet_position]


# ----------------------------------------------------------------------
# initialise the history in a standard way
def add_step_to_history(cs, parent1, parent2, jetp_index, dij):
    max_dij_so_far = max(dij, cs.history[-1].max_dij_so_far)
    cs.history.append(
        HistoryElement(
            parent1=parent1,
            parent2=parent2,
            child=-1,
            jetp_index=jetp_index,
            dij=dij,
            max_dij_so_far=max_dij_so_far,
        )
    )

    local_step = len(cs.history) - 1
    logger.debug(f"Added history step {local_step}: {cs.history[-1]}")

    ##ifndef __NO_ASSERTS__
    # assert(local_step == step_number);
    ##endif

    # sanity check: make sure the particles have not already been recombined
    #
    # Note that good practice would make this an assert (since this is
    # a serious internal issue). However, we decided to throw an
    # InternalError so that the end user can decide to catch it and
    # retry the clustering with a different strategy.

    if parent1 >= 0:
        if cs.history[parent1].child != -1:
            raise (
                RuntimeError(
                    f"Internal error. Trying to recombine a parent1 object that has previsously been recombined: {parent1}"
                )
            )
    cs.history[parent1].child = local_step

    if parent2 >= 0:
        if cs.history[parent2].child != -1:
            raise (
                RuntimeError(
                    f"Internal error. Trying to recombine a parent1 object that has previsously been recombined: {parent2}"
                )
            )
    cs.history[parent2].child = local_step

    # get cross-referencing right from PseudoJets
    if jetp_index >= 0:
        cs.jets[jetp_index].cluster_history_index = local_step


def map_indices_to_julia(tiling_setup, ieta, iphi):
    """Debug mapped indices returning the mapping to a single Julia
    array index, starting from 1"""
    return 1 + ieta + iphi * tiling_setup.n_tiles_eta


def get_tile_indexes(tiling_setup, eta, phi):
    """Return the index of an eta,phi coordinate in the tiling setup"""
    if eta <= tiling_setup.tiles_eta_min:
        ieta = 0
    elif eta >= tiling_setup.tiles_eta_max:
        ieta = tiling_setup.n_tiles_eta - 1
    else:
        ieta = int((eta - tiling_setup.tiles_eta_min) / tiling_setup.tile_size_eta)
        # following needed in case of rare but nasty rounding errors
        if ieta >= tiling_setup.n_tiles_eta:
            ieta = tiling_setup.n_tiles_eta - 1

    # allow for some extent of being beyond range in calculation of phi
    # as well
    # iphi = (int(floor(phi/_tile_size_phi)) + _n_tiles_phi) % _n_tiles_phi;
    # with just int and no floor, things run faster but beware
    # iphi = mod(unsafe_trunc(Int, (phi + 2π) / tiling_setup._tile_size_phi),
    #           tiling_setup._n_tiles_phi)
    iphi = int(phi * tiling_setup.n_tiles_phi / (2.0 * pi))
    return ieta, iphi


def tiledjet_set_jetinfo(jet, cs, jets_index, R2):
    """Setup tiled jet"""
    # Note that tile_indexes is a tuple here, indexing into
    # the Python list of lists
    tile_indexes = get_tile_indexes(cs.tiling.setup, jet.rap, jet.phi)
    tiled_jet = TiledJet(
        id=jets_index,
        eta=jet.rap,
        phi=jet.phi,
        kt2=1.0 / jet.pt2,
        NN_dist=R2,
        jet_index=jets_index,
        tile_index=tile_indexes,
    )
    cs.tiling.tiles[tile_indexes[0]][tile_indexes[1]].append(tiled_jet)
    if len(cs.tiling.tiles[tile_indexes[0]][tile_indexes[1]]) > 1:
        # Do we need this...?
        tiled_jet.previous = cs.tiling.tiles[tile_indexes[0]][tile_indexes[1]][-2]
        cs.tiling.tiles[tile_indexes[0]][tile_indexes[1]][-2].next = tiled_jet
    return tiled_jet


def find_best(diJ, n):
    """Return the value and index of the minimum distance entry"""
    best = 0
    diJ_min = diJ[0]
    for diJ_index, diJ_entry in enumerate(diJ[1:], start=1):
        if diJ_entry < diJ_min:
            best = diJ_index
            diJ_min = diJ_entry
    return diJ_min, best


def find_best_tiledjet(tiledjets: list()):
    """Find the best active jet to merge next"""
    best_jet = -1
    diJ_min = 1e10
    active_jets = 0
    for itiledjet, tiledjet in enumerate(tiledjets):
        if tiledjet.active:
            active_jets += 1
            if tiledjet.diJ_dist < diJ_min:
                best_jet = itiledjet
                diJ_min = tiledjet.diJ_dist
    return diJ_min, best_jet, active_jets


def do_ij_recombination_step(cs, jet_i, jet_j, dij):
    """Carries out the bookkeeping associated with the step of recombining
    jet_i and jet_j (assuming a distance dij) and returns the recombined
    jet and its index position, newjet_k."""
    # Create the new jet by recombining the first two with
    # the E-scheme
    #
    cs.jets.append(cs.jets[jet_i] + cs.jets[jet_j])

    # get its index
    newjet_k = len(cs.jets) - 1

    # get history index
    newstep_k = len(cs.history)

    # and provide jet with the info
    cs.jets[newjet_k].cluster_history_index = newstep_k

    # finally sort out the history
    hist_i = cs.jets[jet_i].cluster_history_index
    hist_j = cs.jets[jet_j].cluster_history_index

    add_step_to_history(cs, min(hist_i, hist_j), max(hist_i, hist_j), newjet_k, dij)

    return cs.jets[-1], newjet_k


def do_iB_recombination_step(cs, jet_i, diB):
    """Carries out the bookkeeping associated with the step of recombining
    jet_i with the beam"""
    # recombine the jet with the beam
    add_step_to_history(cs, cs.jets[jet_i].cluster_history_index, BeamJet, Invalid, diB)


def inclusive_jets(cs, ptmin=0.0):
    """return all inclusive jets of a ClusterSequence with pt > ptmin"""
    dcut = ptmin * ptmin
    jets_local = list()
    # For inclusive jets with a plugin algorithm, we make no
    # assumptions about anything (relation of dij to momenta,
    # ordering of the dij, etc.)
    for elt in reversed(cs.history):
        if elt.parent2 != BeamJet:
            continue
        iparent_jet = cs.history[elt.parent1].jetp_index
        jet = cs.jets[iparent_jet]
        if jet.pt2 >= dcut:
            jets_local.append(jet)

    return jets_local


def faster_tiled_N2_cluster(initial_particles, Rparam=0.4, ptmin=0.0):
    """Tiled AntiKt Jet finding code, implementing the algorithm originally from FastJet"""

    R2 = Rparam * Rparam
    invR2 = 1.0 / R2

    # Create a container of PseudoJet objects
    history, Qtot = initial_history(initial_particles)
    jets = deepcopy(initial_particles)

    # Note that this tiling is filled with blanks - there are no
    # real particles here
    tiling = initial_tiling(initial_particles, Rparam)

    cs = ClusterSequence(jets, history, tiling, Qtot)

    # Not sure we need this - it's a caching object the original code
    tile_union = []

    # Fill basic jet information  to a list of TiledJets
    tiledjets = []
    for ijet, jet in enumerate(jets):
        tiledjets.append(tiledjet_set_jetinfo(jet, cs, ijet, R2))

    # set up the initial nearest neighbour information
    for itilerow, tilerow in enumerate(cs.tiling.tiles):
        for itilecolumn, tile in enumerate(tilerow):
            # In our implementation tile is the list of TiledJets
            # so we can just iterate
            for ijetA, jetA in enumerate(tile, start=1):
                for ijetB, jetB in enumerate(tile[ijetA:], start=ijetA):
                    if jetB == jetA:
                        break
                    dist = tiledjet_dist(jetA, jetB)
                    if dist < jetA.NN_dist:
                        jetA.NN_dist = dist
                        jetA.NN = jetB
                    if dist < jetB.NN_dist:
                        jetB.NN_dist = dist
                        jetB.NN = jetA

            for rightindexes in rightneighbours(itilerow, itilecolumn, tiling.setup):
                neighbourtile = cs.tiling.tiles[rightindexes[0]][rightindexes[1]]
                for jetA in tile:
                    for jetB in neighbourtile:
                        dist = tiledjet_dist(jetA, jetB)
                        if dist < jetA.NN_dist:
                            jetA.NN_dist = dist
                            jetA.NN = jetB
                        if dist < jetB.NN_dist:
                            jetB.NN_dist = dist
                            jetB.NN = jetA

    for tiledjet in tiledjets:
        tiledjet.diJ_dist = tiledjet_diJ(tiledjet)

    # Now run the recombination loop
    history_location = len(cs.jets)
    n = len(cs.jets)
    loop_counter = 0
    while n > 0:
        diJ_min, best, active_jets = find_best_tiledjet(tiledjets)
        loop_counter += 1
        history_location += 1
        n -= 1

        jetA = tiledjets[best]
        jetB = jetA.NN

        # Renormalise
        diJ_min *= invR2
        logger.debug(f"Iteration {loop_counter}: {diJ_min} for jet {jetA.id} and jet {jetB.id if jetA.NN else None}")

        if jetB:
            # jet-jet recombination
            # If necessary relabel A & B to ensure jetB < jetA, that way if
            # the larger of them == newtail then that ends up being jetA and
            # the new jet that is added as jetB is inserted in a position that
            # has a future (probably not relevant in our Python implementation)
            # if jetA.id < jetB.id:
            #     jetA, jetB = jetB, jetA

            # recombine jetA and jetB (adds to the list of TiledJets) and retrieves the new index, nn
            newPseudoJet, nn = do_ij_recombination_step(
                cs, jetA.jet_index, jetB.jet_index, diJ_min
            )

            tiledjet_remove_from_tiles(cs.tiling, jetA)
            jetA.active = False
            tiledjet_remove_from_tiles(cs.tiling, jetB)
            jetB.active = False
            newTiledJet = tiledjet_set_jetinfo(newPseudoJet, cs, nn, R2)
            tiledjets.append(newTiledJet)
        else:
            # jet-beam recombination
            # get the hist_index
            do_iB_recombination_step(cs, jetA.jet_index, diJ_min)
            tiledjet_remove_from_tiles(cs.tiling, jetA)
            jetA.active = False

        # Now establish the set of tiles over which we are going to
        # have to run searches for updated and new nearest-neighbours -
        # basically a combination of vicinity of the tiles of the two old
        # and one new jets.
        tile_union = (
            set()
        )  # Set means we don't need to worry about things being double entered
        add_untagged_neighbours_to_tile_union(jetA.tile_index, tile_union, tiling)
        if jetB and jetB.tile_index != jetA.tile_index:
            add_untagged_neighbours_to_tile_union(jetB.tile_index, tile_union, tiling)
        if (newTiledJet.tile_index != jetA.tile_index) and (
            newTiledJet.tile_index != (jetB.tile_index if jetB else (-1, -1))
        ):
            add_untagged_neighbours_to_tile_union(
                newTiledJet.tile_index, tile_union, tiling
            )

        # Initialise jetB's NN distance as well as updating it for other particles.
        # Run over all tiles in our union
        for tile_index in tile_union:
            try:
                tile = tiling.tiles[tile_index[0]][tile_index[1]]
            except IndexError as e:
                print(f"{tile_index} caused {e}")
                exit(1)

            # run over all jets in the current tile
            for jetI in tile:
                # see if jetI had jetA or jetB as a NN -- if so recalculate the NN
                if (jetI.NN == jetA) or (jetI.NN == jetB):
                    # print("Strike!")
                    jetI.NN_dist = R2
                    jetI.NN = None

                    # now go over tiles that are neighbours of this jet (include own tile)
                    for near_tile_index in surrounding_tiles(tile_index, tiling):
                        # and then over the contents of that tile
                        for jetJ in tiling.tiles[near_tile_index[0]][
                            near_tile_index[1]
                        ]:
                            dist = tiledjet_dist(jetI, jetJ)
                            if (dist < jetI.NN_dist) and (jetJ.id != jetI.id):
                                jetI.NN_dist = dist
                                jetI.NN = jetJ
                    jetI.diJ_dist = tiledjet_diJ(jetI)

                # check whether new newTiledJet is closer than jetI's current NN and
                # if jetI is closer than newTiledJet's current (evolving) nearest
                # neighbour. Where relevant update things.
                if newTiledJet.isvalid():
                    dist = tiledjet_dist(jetI, newTiledJet)
                    if dist < jetI.NN_dist:
                        if jetI != newTiledJet:
                            jetI.NN_dist = dist
                            jetI.NN = newTiledJet
                            jetI.diJ_dist = tiledjet_diJ(jetI)
                    if (dist < newTiledJet.NN_dist) and (jetI != newTiledJet):
                        newTiledJet.NN_dist = dist
                        newTiledJet.NN = jetI
        if newTiledJet.isvalid():
            newTiledJet.diJ_dist = tiledjet_diJ(newTiledJet)

    return inclusive_jets(cs, ptmin)
