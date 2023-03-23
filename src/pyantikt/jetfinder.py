from copy import deepcopy
from math import pi, floor, trunc
from sys import float_info


class TilingDef:
    """Simple class with tiling parameters"""

    def __init__(
        self,
        tiles_eta_min,
        tiles_eta_max,
        tile_size_eta,
        tile_size_phi,
        n_tiles_eta,
        n_tiles_phi,
        tiles_ieta_min,
        tiles_ieta_max,
    ):
        self.tiles_eta_min = tiles_eta_min
        self.tiles_eta_max = tiles_eta_max
        self.tile_size_eta = tile_size_eta
        self.tile_size_phi = tile_size_phi
        self.n_tiles_eta = n_tiles_eta
        self.n_tiles_phi = n_tiles_phi
        self.n_tiles = self.n_tiles_eta * self.n_tiles_phi
        self.tiles_ieta_min = tiles_ieta_min
        self.tiles_ieta_max = tiles_ieta_max


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

    # start scan over rapidity bins from the left, to find out minimum rapidity of tiling
    cumul_lo = 0.0
    cumul2 = 0.0
    ibin_lo = 0
    while ibin_lo < nbins:
        cumul_lo += counts[ibin_lo]
        if cumul_lo >= allowed_max_cumul:
            minrap = max(minrap, ibin_lo - nrap)
            print(minrap, ibin_lo - nrap)
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

    # print(minrap, maxrap, ibin_lo, ibin_hi, cumul2)

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
    # Tiling(tiling_setup)


def faster_tiled_N2_cluster(initial_particles, Rparam=0.4, ptmin=0.0):
    """Tiled AntiKt Jet finding code, implementing the algorithm originally from FastJet"""

    R2 = Rparam * Rparam
    invR2 = 1.0 / R2

    # Create a container of PseudoJet objects
    jets = deepcopy(initial_particles)
    tiling = initial_tiling(initial_particles, Rparam)
