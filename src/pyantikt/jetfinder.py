from copy import deepcopy
from math import pi, floor, trunc
from sys import float_info, exit
from pyantikt.tiles import TilingDef, TiledJet, Tiling
from pyantikt.history import HistoryElement, ClusterSequence


def tiledjet_dist(jetA: TiledJet, jetB: TiledJet):
    '''Distance between two tiled jets'''
    dphi = pi - abs(pi - abs(jetA.phi - jetB.phi))
    deta = jetA.eta - jetB.eta
    return dphi*dphi + deta*deta


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

    # print(minrap, maxrap, counts)

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
            # print(minrap, ibin_lo - nrap)
            break
        # print(cumul_lo, ibin_lo)
        ibin_lo += 1
    if ibin_lo == nbins:
        raise RuntimeError(
            "Failed to find a low bin"
        )  # internal consistency check that you found a bin
    cumul2 += cumul_lo**2
    # print("c2 lo", cumul2)

    # then do it from right, to find out maximum rapidity of tiling
    cumul_hi = 0.0
    ibin_hi = nbins - 1
    while ibin_hi >= 0:
        cumul_hi += counts[ibin_hi]
        if cumul_hi >= allowed_max_cumul:
            maxrap = min(maxrap, ibin_hi - nrap + 1)
            break
        # print(cumul_hi, ibin_hi)
        ibin_hi -= 1
    if ibin_hi == -1:
        raise RuntimeError(
            "Failed to find a high bin"
        )  # internal consistency check that you found a bin

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
        # print("c2 hi", cumul2)

    # now get the rest of the squared bin contents
    for ibin in range(ibin_lo + 1, ibin_hi + 1):
        cumul2 += counts[ibin] ** 2
        # print(ibin, cumul2)

    print(minrap, maxrap, ibin_lo, ibin_hi, cumul2)

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
    print(f"Tiling is {n_tiles_eta} x {n_tiles_phi} in eta, phi")
    return Tiling(tiling_setup)


def initial_history(particles):
    """Initialise the clustering history in a standard way,
    Takes as input the list of stable particles as input
    Returns the history and the total event energy."""

    # This is going to be a list of HistoryElements
    history = []
    Qtot = 0.0

    for i, _ in enumerate(particles):
        # Add in order, so that HistoryElement[i] -> particle[i]
        history.append(HistoryElement(jetp_index=i))

        # get cross-referencing right from PseudoJets
        particles[i].cluster_hist_index = i

        # determine the total energy in the event
        Qtot += particles[i].E

    return history, Qtot

def map_indices_to_julia(tiling_setup, ieta, iphi):
    '''Debug mapped indices returning the mapping to a single Julia
    array index, starting from 1'''
    return 1 + ieta + iphi * tiling_setup.n_tiles_eta

def get_tile_indexes(tiling_setup, eta, phi):
    '''Return the index of an eta,phi coordinate in the tiling setup'''
    if eta <= tiling_setup.tiles_eta_min:
        ieta = 0
    elif eta >= tiling_setup.tiles_eta_max:
        ieta = tiling_setup.n_tiles_eta-1
    else:
        ieta = int((eta - tiling_setup.tiles_eta_min) / tiling_setup.tile_size_eta)
        # following needed in case of rare but nasty rounding errors
        if ieta >= tiling_setup.n_tiles_eta:
            ieta = tiling_setup.n_tiles_eta-1

    # allow for some extent of being beyond range in calculation of phi
    # as well
    #iphi = (int(floor(phi/_tile_size_phi)) + _n_tiles_phi) % _n_tiles_phi;
    # with just int and no floor, things run faster but beware
    #iphi = mod(unsafe_trunc(Int, (phi + 2π) / tiling_setup._tile_size_phi),
    #           tiling_setup._n_tiles_phi)
    iphi = int(phi * tiling_setup.n_tiles_phi / (2.0 * pi))
    print(ieta, iphi, map_indices_to_julia(tiling_setup, ieta, iphi))
    return ieta, iphi

def tiledjet_set_jetinfo(jet, cs, jets_index, R2):
    '''Setup tiled jet'''
    tile_indexes = get_tile_indexes(cs.tiling.setup, jet.rap, jet.phi)
    tiled_jet = TiledJet(id=jets_index,
                                eta=jet.rap,
                                phi=jet.phi,
                                kt2=1.0/jet.pt2,
                                NN_dist=R2,
                                jet_index=jets_index,
                                tile_index=tile_indexes
                                )
    cs.tiling.tiles[tile_indexes[0]][tile_indexes[1]].append(tiled_jet)
    if len(cs.tiling.tiles[tile_indexes[0]][tile_indexes[1]]) > 1:
        # Do we need this...?
        tiled_jet.previous = cs.tiling.tiles[tile_indexes[0]][tile_indexes[1]][-2]
        cs.tiling.tiles[tile_indexes[0]][tile_indexes[1]][-2].next = tiled_jet
    return tiled_jet

def faster_tiled_N2_cluster(initial_particles, Rparam=0.4, ptmin=0.0):
    """Tiled AntiKt Jet finding code, implementing the algorithm originally from FastJet"""

    R2 = Rparam * Rparam
    invR2 = 1.0 / R2

    # Create a container of PseudoJet objects
    jets = deepcopy(initial_particles)
    history, Qtot = initial_history(initial_particles)
    # print(history, Qtot)

    # Note that this tiling is filled with blanks - there are no
    # real particles here
    tiling = initial_tiling(initial_particles, Rparam)
    # print(tiling)

    cs = ClusterSequence(jets, history, tiling, Qtot)
    # print(cs.tiling.tiles)
    # print(len(cs.tiling.tiles))
    # for slice in cs.tiling.tiles:
    #     print(len(slice))
    # exit(0)



    # Not sure we need this - it's a caching object the original code
    tile_union = []

    # Fill basic jet information  to a list of TiledJets
    tiledjets = []
    for ijet, jet in enumerate(jets):
        tiledjets.append(tiledjet_set_jetinfo(jet, cs, ijet, R2))
        # print(ijet, tiledjets[-1])

    print(len(tiledjets))

    # set up the initial nearest neighbour information
    for tilerow in cs.tiling.tiles:
        for tile in tilerow:
            # In our implementation tile is the list of TiledJets
            # so we can just iterate
            print(f"Start tile length {len(tile)}")
            for ijetA, jetA in enumerate(tile, start=1):
                for ijetB, jetB in enumerate(tile[ijetA:], start=ijetA):
                    print(f"A, B: {ijetA-1}, {ijetB}")
                    if jetB == jetA:
                        break
                    dist = tiledjet_dist(jetA, jetB)
                    print(dist)
                    if (dist < jetA.NN_dist):
                        jetA.NN_dist = dist
                        jetA.NN = jetB
                    if dist < jetB.NN_dist:
                        jetB.NN_dist = dist
                        jetB.NN = jetA

    exit(0)
