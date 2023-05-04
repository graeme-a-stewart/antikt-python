#! /usr/bin/env python3

import json
import unittest
import time

from pprint import pprint

from pyantikt.hepmc import read_jet_particles
from pyantikt.acceleratedbasicjetfinder import basicjetfinder


def find_python_jets(nevents: int):
    """Separate our jet finding to avoid running multiple times"""
    # Read events from HEPMC file and find our jets
    events = read_jet_particles(
        "../data/events.hepmc3",
        skip=0,
        nevents=nevents,
    )
    my_jets = []

    # N.B. Match Julia's natural counting here, start at 1!
    print("Event: ", end="")
    for ievt, event in enumerate(events, start=1):
        print(".", end="", flush=True)
        antikt_jets = basicjetfinder(event, Rparam=0.4, ptmin=5.0)
        jet_list = []
        for ijet, jet in enumerate(antikt_jets):
            jet_list.append({"rap": jet.rap, "phi": jet.phi, "pt": jet.pt})
        my_jets.append({"jetid": ievt, "jets": jet_list})
    print()

    return my_jets


def get_reference_jets(source_file):
    # Read reference jet outputs
    with open(source_file) as refjets:
        return json.load(refjets)


class TestJetOutputs(unittest.TestCase):
    # Set these up as class variables, so they are only initialised once
    reference_jets_fastjet = None
    reference_jets_julia = None
    my_jets = None

    # As our jet finding is expensive, only do this once
    @classmethod
    def setUpClass(self) -> None:
        self.reference_jets_fastjet = get_reference_jets(
            "../data/jet_collections_fastjet.json"
        )
        self.my_jets = find_python_jets(len(self.reference_jets_fastjet))

    def test_jet_outputs_fastjet(self):
        for ievt, (ref_evt, my_evt) in enumerate(
            zip(self.reference_jets_fastjet, self.my_jets), start=1
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
