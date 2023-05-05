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
    parser.add_argument("--eventnumber", action="store_true", default=False,
                        help="Also write the event number to the output file")
    parser.add_argument("eventfile", help="File with HepMC3 events to process")
    parser.add_argument("output", metavar="FILE", help="Write event output to FILE")

    args = parser.parse_args(sys.argv[1:])

    events = read_jet_particles(
        file=args.eventfile, skip=args.skip, nevents=args.maxevents
    )

    if len(events) > 1 and not args.eventnumber:
        print("Warning, multiple events will be written to the same file without eventnumbers")

    if args.stats:
        statsf = open(args.stats, "w")
        print("Input_Particles", file=statsf)
    with open(args.output, "w") as outf:
        for ievent, event in enumerate(events):
            for particle in event:
                if args.eventnumber:
                    print(f"{ievent} ", file=outf, end="")
                print(f"{particle.px} {particle.py} {particle.pz} {particle.E}", file=outf)
            if (args.stats):
                print(len(event), file=statsf)

if __name__ == "__main__":
    main()
