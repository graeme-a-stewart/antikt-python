# Classes for history and sequences

from dataclasses import dataclass
from pyantikt.pseudojet import PseudoJet
from pyantikt.tiles import Tiling


@dataclass
class HistoryElement:
    # Index in _history where first parent of this jet
    # was created (-1 if this jet is an
    # original particle)
    parent1: int = -1

    # index in _history where second parent of this jet
    # was created (-1 if this jet is an
    # original particle); BeamJet if this history entry
    # just labels the fact that the jet has recombined
    # with the beam)
    parent2: int = -1

    # index in _history where the current jet is
    # recombined with another jet to form its child. It
    # is -1 if this jet does not further
    # recombine.
    child: int = -1

    # index in the _jets vector where we will find the
    # PseudoJet object corresponding to this jet
    # (i.e. the jet created at this entry of the
    # history). NB: if this element of the history
    # corresponds to a beam recombination, then
    # jetp_index=Invalid.
    jetp_index: int = -1

    # the distance corresponding to the recombination
    # at this stage of the clustering.
    dij: float = 0.0

    # the largest recombination distance seen
    # so far in the clustering history.
    max_dij_so_far: float = 0.0


@dataclass
class ClusterSequence:
    # This contains the physical PseudoJets; for each PseudoJet one
    # can find the corresponding position in the _history by looking
    # at _jets[i].cluster_hist_index()
    jets: list[PseudoJet]

    # This vector will contain the branching history; for each stage,
    # _history[i].jetp_index indicates where to look in the _jets
    # vector to get the physical PseudoJet."""
    history: list[HistoryElement]

    # PseudoJet tiling
    tiling: Tiling

    # Total energy of the event
    Qtot: float = 0.0
