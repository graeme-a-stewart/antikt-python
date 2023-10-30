# FastJet Test Application

This code compiles a small FastJet application that is used for benchmarking and
validation of alternative implementations of clustering algorithms.

The code requires the fastjet libraries (<https://fastjet.fr/>) as well as those
from HepMC3 (<https://gitlab.cern.ch/hepmc/HepMC3>).

Once FastJet and HepMC3 are available compile `chep-polyglot-jets.cc` and link
against the relevant libraries. See the `Makefile` for an indication of more or
less how to do that.

Run the application with no arguments to understand the command line parameters.

Running like this:

```sh
./chep-polyglot-jets ../data/events.hepmc3 100 100 N2Tiled   
```

Would benchmark the N2Tiled strategy against the 100 *benchmark* events, running
100 times.

```sh
./chep-polyglot-jets ../data/events.hepmc3 100 1 N2Tiled 1 1 > inclusivekt.out
```

Would use the *inclusive kt* algorithm ($p=1$) and dump the final jet
constituents to `inclusivekt.out`.

Then the script `fastjet-output2json.py` can be used to process this into JSON
that can be used more easily in tests to check consistency between FastJet and
other implementations.
