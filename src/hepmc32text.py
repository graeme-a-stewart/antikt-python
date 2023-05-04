#! /usr/bin/env python3

"""Convert HepMC3 event(s) to a simple list of px, py, pz E"""

import argparse
import sys

from pyantikt.hepmc import read_jet_particles

def main():
    parser = argparse.ArgumentParser(description="AntiKt Basic Jet Finder")
    parser.add_argument(
        "--skip", type=int, default=0, help="Number of input events to skip"
    )
    parser.add_argument(
        "--maxevents",
        "-n",
        type=int,
        default=1,
        help="Maximum number of events to process",
    )
    parser.add_argument("--stats", help="Dump stats of number of particles per event to file")
    parser.add_argument("eventfile", help="File with HepMC3 events to process")
    parser.add_argument("output", metavar="FILE", help="Write event output to FILE")

    args = parser.parse_args(sys.argv[1:])

    events = read_jet_particles(
        file=args.eventfile, skip=args.skip, nevents=args.maxevents
    )

    if args.stats:
        statsf = open(args.stats, "w")
    for ievent, event in enumerate(events):
        if args.maxevents > 1:
            filename = f"{args.output}-{ievent:02}.txt"
        else:
            filename = args.output
        
        with open(filename, "w") as outf:
            for particle in event:
                print(f"{particle.px} {particle.py} {particle.pz} {particle.E}", file=outf)
        if (args.stats):
            print(len(event), file=statsf)

if __name__ == "__main__":
    main()
