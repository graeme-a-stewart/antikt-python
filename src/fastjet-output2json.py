#! /usr/bin/env python3

"""Convert stdout from the fastjet timing run to JSON for
comparison with Julia and Python output"""

import argparse
import json
from pprint import pprint
import re
import sys


def main():
    parser=argparse.ArgumentParser(description="Convert fastjet output to JSON")
    parser.add_argument("input", help="Fastjet chep-polyglot-jets output")
    parser.add_argument("output", help="JSON output filename")

    args = parser.parse_args(sys.argv[1:])

    fastjet_events = []
    event_number = 0
    jets = []
    with open(args.input) as fastjet_input:
        for line in fastjet_input:
            if line.startswith("#"):
                continue
            if re.match(r"^\s$", line):
                continue
            if line.startswith("Jets in"):
                # New event header
                # First, do we currently have an event?
                if event_number > 0 and len(jets) > 0:
                    new_event = {"jetid": event_number, "jets": jets}
                    fastjet_events.append(new_event)
                    jets = []
                # Find event number
                match = re.search(r"(\d+)", line)
                if match:
                    # Use Julia numbering, from 1, should be in the input
                    event_number = int(match.group(1))
                else:
                    raise RuntimeError(f"Failed to find event number in {line}")
            else:
                # Jet contents are prepended by spaces, everything else is irrelevant
                # (Fastjet headers, final stats)
                if not line.startswith(" "):
                    continue
                line = line.strip()
                # Should be a jet...
                # print(line.split())
                (ijet, rap, phi, pt) = line.split()
                ijet = int(ijet)
                rap = float(rap)
                phi = float(phi)
                pt = float(pt)
                jets.append({"rap": rap, "phi": phi, "pt": pt})
    # Process the last dangling event...
    new_event = {"jetid": event_number, "jets": jets}
    fastjet_events.append(new_event)
    # pprint(fastjet_events)

    with open(args.output, "w") as json_out:
        json.dump(fastjet_events, json_out, indent=2)


if __name__ == "__main__":
    main()
