#! /usr/bin/env python3
"""Anti-Kt jet finder, Tiled N^2 version"""

import argparse
import numpy as np
import vector
import sys
import logging

from copy import deepcopy
from pathlib import Path

from pyantikt.hepmc import read_jet_particles
from pyantikt.jetfinder import faster_tiled_N2_cluster

def main():
    parser = argparse.ArgumentParser(description="Tiled N^2 AntiKt Jet Finder")
    parser.add_argument("--skip", type=int,
                        default=0, help="Number of input events to skip")
    parser.add_argument("--maxevents", "-n", type=int,
                        default=1, help="Maximum number of events to process")
    parser.add_argument("--debug", action='store_true',
                        help="Activate logging debugging mode")
    parser.add_argument("--info", action='store_true',
                        help="Activate logging info mode")
    parser.add_argument("eventfile",
                        help="File with HepMC3 events to process")
    
    args = parser.parse_args(sys.argv[1:])

    if args.info:
        logger.setLevel(logging.INFO)
    if args.debug:
        logger.setLevel(logging.DEBUG)

    events = read_jet_particles(file=args.eventfile, skip=args.skip, nevents=args.maxevents)
    logger.info(events[0][0])

    for event in events:
        antikt_jets = faster_tiled_N2_cluster(event, Rparam=0.4, ptmin=5.0)
        print(f"Found {len(antikt_jets)} jets")
        for ijet, jet in enumerate(antikt_jets):
            print(ijet, jet.rap, jet.phi, jet.pt)
            

if __name__ == "__main__":
    logger = logging.getLogger(Path(sys.argv[0]).name)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(name)s:%(levelname)s %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    main()
