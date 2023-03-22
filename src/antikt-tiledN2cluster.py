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

    initial_particles = read_jet_particles()
    logger.info(initial_particles[0])
    logger.debug("It's cheese time!")
    print(initial_particles)

    antikt_jet = faster_tiled_N2_cluster(initial_particles)


if __name__ == "__main__":
    logger = logging.getLogger(Path(sys.argv[0]).name)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(name)s:%(levelname)s %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    main()
