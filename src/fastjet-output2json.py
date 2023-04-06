#! /usr/bin/env python3

'''Convert stdout from the fastjet timing run to JSON for
comparison with Julia and Python output'''

import json
from pprint import pprint
import re

def main():
    fastjet_events = []
    event_number = 0
    jets = []
    with open("../data/fastjet.out") as fastjet_input:
        for line in fastjet_input:
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith("#"):
                continue
            if line.startswith("Jets in"):
                # New event header
                # First, do we currently have an event?
                if event_number > 0 and len(jets) > 0:
                    new_event = {'jetid': event_number,
                                 'jets': jets}
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
                # Should be a jet...
                print(line.split())
                (ijet, rap, phi, pt) = line.split()
                ijet = int(ijet)
                rap = float(rap)
                phi = float(phi)
                pt = float(pt)
                jets.append({'rap': rap,
                             'phi': phi,
                             'pt': pt})
    # Process the last dangling event...
    new_event = {'jetid': event_number, 'jets': jets}
    fastjet_events.append(new_event)
    # pprint(fastjet_events)

    with open("../data/jet_collections_fastjet.json", "w") as json_out:
        json.dump(fastjet_events, json_out, indent=2)

if __name__ == "__main__":
    main()
