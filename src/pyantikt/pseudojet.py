from math import atan2, pi, log


# A few saftey factor constants
_MaxRap = 1e5

# _invalid_phi = -100.0
# _invalid_rap = -1.0e200


class PseudoJet:
    def __init__(self, px, py, pz, E):
        self._px = px
        self._py = py
        self._pz = pz
        self._E = E

        self._pt2 = px * px + py * py
        self._inv_pt2 = 1.0 / self._pt2

        self._rap = self._set_rap()
        self._phi = self._set_phi()

        self._cluster_history_index = -1

    def _set_rap(self):
        if (self._E == abs(self._pz)) and (self._pt2 == 0.0):
            # Point has infinite rapidity -- convert that into a very large
            #    number, but in such a way that different 0-pt momenta will have
            #    different rapidities (so as to lift the degeneracy between
            #                         them) [this can be relevant at parton-level]
            MaxRapHere = _MaxRap + abs(self._pz)
            return MaxRapHere if p.pz >= 0.0 else -MaxRapHere
        effective_m2 = max(0.0, self.m2)  # force non tachyonic mass
        E_plus_pz = self._E + abs(self._pz)  # the safer of p+, p-
        rapidity = 0.5 * log((self._pt2 + effective_m2) / (E_plus_pz * E_plus_pz))
        return rapidity if self._pz < 0 else -rapidity

    def _set_phi(self):
        if self._pt2 == 0.0:
            phi = 0.0
        else:
            phi = atan2(self._py, self._px)
        if phi < 0.0:
            phi += 2.0 * pi
        elif phi > 2.0 * pi:
            phi -= 2.0 * pi
        return phi

    def __str__(self):
        return (
            f"PseudoJet (px: {self._px}, py: {self._py}, pz: {self._pz}, E: {self._E})"
        )

    @property
    def m2(self):
        """squared invariant mass"""
        return (self._E + self._pz) * (self._E - self._pz) - self._pt2

    @property
    def px(self):
        return self._px

    @property
    def py(self):
        return self._py

    @property
    def pz(self):
        return self._pz

    @property
    def E(self):
        return self._E

    @property
    def rap(self):
        return self._rap

    @property
    def phi(self):
        return self._phi

    @property
    def phi(self):
        return self._phi

    @property
    def pt2(self):
        return self._pt2

    @property
    def inv_pt2(self):
        return self._inv_pt2
