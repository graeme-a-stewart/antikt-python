#! /usr/bin/env python3

import json
import unittest

from pprint import pprint

from pyantikt.hepmc import read_jet_particles
from pyantikt.jetfinder import faster_tiled_N2_cluster


class TestJetOutputs(unittest.TestCase):
    def setUp(self):
        # Read reference jet outputs
        with open("../data/jet_collections.json") as refjets:
            self.reference_jets = json.load(refjets)

        # Read events from HEPMC file and find our jets
        events = read_jet_particles(
            "../data/events.hepmc3/events.hepmc3",
            skip=0,
            nevents=len(self.reference_jets),
        )
        self.my_jets = []

        # N.B. Match Julia's natural counting here, start at 1!
        for ievt, event in enumerate(events, start=1):
            antikt_jets = faster_tiled_N2_cluster(event, Rparam=0.4, ptmin=5.0)
            jet_list = []
            for ijet, jet in enumerate(antikt_jets):
                jet_list.append({"rap": jet.rap, "phi": jet.phi, "pt": jet.pt})
            self.my_jets.append({"jetid": ievt, "jets": jet_list})

    def test_jet_outputs(self):
        for ievt, (ref_evt, my_evt) in enumerate(
            zip(self.reference_jets, self.my_jets), start=1
        ):
            # Consider each event as a subtest, which means we don't stop for
            # failed events, but carry on to test the remaining events
            with self.subTest(ievt=ievt, ref_evt=ref_evt, my_evt=my_evt):
                self.assertEqual(
                    len(ref_evt["jets"]),
                    len(my_evt["jets"]),
                    f"Inconsistent number of jets for event {ievt}",
                )
                # Just to be safe, sort the list of jets by pt
                ref_evt["jets"].sort(key=lambda jet: jet["pt"])
                my_evt["jets"].sort(key=lambda jet: jet["pt"])
                for ijet, (ref_jet, my_jet) in enumerate(
                    zip(ref_evt["jets"], my_evt["jets"]), start=1
                ):
                    self.assertAlmostEqual(
                        ref_jet["pt"],
                        my_jet["pt"],
                        msg=f"pt does not match for event {ievt}, sorted jet {ijet}",
                    )
                    self.assertAlmostEqual(
                        ref_jet["rap"],
                        my_jet["rap"],
                        msg=f"rapidity does not match for event {ievt}, sorted jet {ijet}",
                    )
                    self.assertAlmostEqual(
                        ref_jet["phi"],
                        my_jet["phi"],
                        msg=f"phi does not match for event {ievt}, sorted jet {ijet}",
                    )


if __name__ == "__main__":
    unittest.main()
