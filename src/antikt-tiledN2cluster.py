#! /usr/bin/env python3
"""Anti-Kt jet finder, Tiled N^2 version
   Can use numba and numpy for acceleration
"""

import argparse
import sys
import time
import logging

from copy import deepcopy
from pathlib import Path

from pyantikt.hepmc import read_jet_particles

from pyantikt.benchmark import Benchmark

try:
    import pyantikt.acceleratedtiledjetfinder
except ImportError as e:
    print(f"pyantikt.acceleratedtiledjetfinder unavailable: {e}")
import pyantikt.tiledjetfinder


def main():
    parser = argparse.ArgumentParser(description="Tiled N^2 AntiKt Jet Finder")
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
    parser.add_argument(
        "--trials", type=int, default=1, help="Number of trials to repeat"
    )
    parser.add_argument("--output", metavar="FILE", help="Write logging output to FILE")
    parser.add_argument(
        "--numba", action="store_true", help="Run accelerated numba code version"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Activate logging debugging mode"
    )
    parser.add_argument(
        "--info", action="store_true", help="Activate logging info mode"
    )
    parser.add_argument("--benchmark", help="Benchmark results to a file")
    parser.add_argument("eventfile", help="File with HepMC3 events to process")

    args = parser.parse_args(sys.argv[1:])

    if args.info:
        logger.setLevel(logging.INFO)
        logging.getLogger("jetfinder").setLevel(logging.INFO)
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("jetfinder").setLevel(logging.DEBUG)

    if args.output:
        logout = logging.FileHandler(args.output)
        logging.getLogger("jetfinder").addHandler(logout)

    # Switch between implenentations here
    if args.numba:
        try:
            faster_tiled_N2_cluster = (
                pyantikt.acceleratedtiledjetfinder.faster_tiled_N2_cluster
            )
        except AttributeError as e:
            raise RuntimeError(
                "Numba accelerated code requested, but it's unavailable"
            ) from e
    else:
        faster_tiled_N2_cluster = pyantikt.tiledjetfinder.faster_tiled_N2_cluster

    original_events = read_jet_particles(
        file=args.eventfile, skip=args.skip, nevents=args.maxevents
    )

    benchmark = Benchmark(nevents=args.maxevents)

    # If we are bencmarking the numba code, do a warm up run
    # to jit compile the accelerated code
    if args.trials > 1 and args.numba:
        print("Warm up run with first event to jit compile code")
        faster_tiled_N2_cluster(deepcopy(original_events[0]), Rparam=0.4, ptmin=0.5)

    for itrial in range(1, args.trials + 1):
        if args.trials > 1:
            events = deepcopy(original_events)
        else:
            events = original_events
        start = time.monotonic_ns() / 1000.0  # microseconds
        for ievt, event in enumerate(events, start=1):
            logger.info(f"Event {ievt} has {len(event)} particles")
            antikt_jets = faster_tiled_N2_cluster(event, Rparam=0.4, ptmin=5.0)
            logger.info(f"Event {ievt}, found {len(antikt_jets)} jets")
            for ijet, jet in enumerate(antikt_jets):
                logger.info(f"{ijet}, {jet.rap}, {jet.phi}, {jet.pt}")
        end = time.monotonic_ns() / 1000.0
        benchmark.runtimes.append(end - start)
        print(f"Trial {itrial}. Processed {len(events)} events in {end-start:,.2f} us")
        print(f"Time per event: {(end-start)/len(events):,.2f} us")
    if args.trials > 1:
        mean, stddev = benchmark.get_stats()
        print(f"Mean time per event {mean:,.2f} ± {stddev:,.2f} us")
        minimum = benchmark.get_minimum()
        print(f"Minimum time per event {minimum:,.2f} us")

    if args.benchmark:
        with open(args.benchmark, mode="w") as benchmark_file:
            print(benchmark.to_json(), file=benchmark_file)
        logger.info(benchmark)


if __name__ == "__main__":
    logger = logging.getLogger(Path(sys.argv[0]).name)
    logger.setLevel(logging.WARN)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(name)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logging.getLogger("jetfinder").addHandler(ch)

    main()
