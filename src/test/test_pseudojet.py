#! /usr/bin/env python3

import unittest
import vector
from pyantikt.pseudojet import pseudojet


class TestDerivedLorentzProperties(unittest.TestCase):
    def setUp(self):
        px = 0.22337332139323995
        py = 0.013200084794337135
        pz = -0.18261482717845642
        E = 0.28882184483560597

        self.vector_pseudojet = vector.obj(px=px, py=py, pz=pz, E=E)

        self.pyjet_pesudojet = pseudojet(px=px, py=py, pz=pz, E=E)

    def test_rapidity(self):
        self.assertAlmostEqual(self.vector_pseudojet.rapidity, self.pyjet_pesudojet.rap)

    def test_phi(self):
        self.assertAlmostEqual(self.vector_pseudojet.phi, self.pyjet_pesudojet.phi)

    def test_pt2(self):
        self.assertAlmostEqual(self.vector_pseudojet.pt2, self.pyjet_pesudojet.pt2)


if __name__ == "__main__":
    unittest.main()
