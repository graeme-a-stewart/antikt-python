from math import atan2, pi, log, sqrt


# A few saftey factor constants
_MaxRap = 1e5

# _invalid_phi = -100.0
# _invalid_rap = -1.0e200


class PseudoJet:
    def __init__(self, px, py, pz, E):
        self.px = px
        self.py = py
        self.pz = pz
        self.E = E

        self.pt2 = px * px + py * py
        self.inv_pt2 = 1.0 / self.pt2

        self.rap = self._set_rap()
        self.phi = self._set_phi()

        self.cluster_history_index = -1

    def _set_rap(self):
        if (self.E == abs(self.pz)) and (self.pt2 == 0.0):
            # Point has infinite rapidity -- convert that into a very large
            #    number, but in such a way that different 0-pt momenta will have
            #    different rapidities (so as to lift the degeneracy between
            #                         them) [this can be relevant at parton-level]
            MaxRapHere = _MaxRap + abs(self.pz)
            return MaxRapHere if self.pz >= 0.0 else -MaxRapHere
        effective_m2 = max(0.0, self.m2)  # force non tachyonic mass
        E_plus_pz = self.E + abs(self.pz)  # the safer of p+, p-
        rapidity = 0.5 * log((self.pt2 + effective_m2) / (E_plus_pz * E_plus_pz))
        return rapidity if self.pz < 0 else -rapidity

    def _set_phi(self):
        if self.pt2 == 0.0:
            phi = 0.0
        else:
            phi = atan2(self.py, self.px)
        if phi < 0.0:
            phi += 2.0 * pi
        elif phi > 2.0 * pi:
            phi -= 2.0 * pi
        return phi

    def __str__(self):
        return (
            f"PseudoJet (px: {self.px}, py: {self.py}, pz: {self.pz}, E: {self.E})"
        )

    @property
    def pt(self):
        """transverse momentum"""
        return sqrt(self.pt2)

    @property
    def m2(self):
        """squared invariant mass"""
        return (self.E + self.pz) * (self.E - self.pz) - self.pt2

    # Need to define the + operator on two jets
    def __add__(self, jetB):
        px = self.px + jetB.px
        py = self.py + jetB.py
        pz = self.pz + jetB.pz
        E = self.E + jetB.E
        return PseudoJet(px, py, pz, E)
