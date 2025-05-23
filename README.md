# AntiKt-Python

This package is an implementation of the [AntiKt algorithm](https://arxiv.org/abs/0802.1189) in Python, with both a basic version and a tiled implementation.

There is a pure Python implementation and one using `numpy` plus `numba`.

This code is used in the CHEP 2023 Paper, [*Polyglot Jet Finding*](https://doi.org/10.1051/epjconf/202429505017).

## HOWTO

This code has been tested and should run fine in Python 3.10 to at least 3.13.

### Environment

Use the `environment.yml` file to setup the environment (you may wish to edit the main Python version first) with conda/mamba.

### Running

There are two Python "main" source files that implement the Anti-Kt sequential jet finding algorithm:

| Anti-Kt Strategy | File |
|---|---|
| `antikt-basic.py` | `N2Plain` |
| `antikt-tiledN2cluster.py` | `N2Tiled` |

Please see the documentation of either [Fastjet](https://fastjer.fr/) or [JetReconstruction.jl](https://juliahep.github.io/JetReconstruction.jl/stable/) for a description of their characteristics.

For both of these strategies there are two *implementations*, one (default) in pure Python, and another using accelerated Python, with Numba and Numpy. The latter is enabled by adding the `--numba` switch.

Running the code it quite straight forward:

```sh
$ ./antikt-basic.py --maxevents=100 --trials 4 --radius 1.0 ../data/events.hepmc3
Trial 1. Processed 100 events in 8,862,307.08 us
Time per event: 88,623.07 us
Trial 2. Processed 100 events in 8,861,340.71 us
Time per event: 88,613.41 us
Trial 3. Processed 100 events in 8,817,813.50 us
Time per event: 88,178.13 us
Trial 4. Processed 100 events in 8,989,259.50 us
Time per event: 89,892.60 us
Mean time per event 88,826.80 ± 740.21 us
Minimum time per event 88,178.13 us
```

```sh
./antikt-tiledN2cluster.py --maxevents=100 --trials 4 --numba ../data/events.hepmc3
Warm up run with first event to jit compile code
Trial 1. Processed 100 events in 1,807,147.96 us
Time per event: 18,071.48 us
Trial 2. Processed 100 events in 1,787,033.62 us
Time per event: 17,870.34 us
Trial 3. Processed 100 events in 1,773,394.25 us
Time per event: 17,733.94 us
Trial 4. Processed 100 events in 1,788,132.88 us
Time per event: 17,881.33 us
Mean time per event 17,889.27 ± 138.74 us
Minimum time per event 17,733.94 us
```

The "standard" source file `events.hepmc3` contains 100 13TeV pp events, generated by Pythia. Please see the [JetReconstructionBenchmarks.jl](https://github.com/graeme-a-stewart/JetReconstructionBenchmarks.jl) repository for different event input files, with different intitial particle densities.

## Copyright

All files are Copyright (C) CERN, 2023-2025 and provided under the Apache-2 license.

## Acknowledgements

The antikt N^2 tiled algorithm is adapted from ClusterSequence_Tiled_N2.cc c++ code 
from the Fastjet (<https://fastjet.fr>,  hep-ph/0512210,  arXiv:1111.6097).

The reimplementation of the algorithm by Philippe Gras in Julia was extremely
useful when developing this Python version (<https://github.com/grasph/AntiKt.jl>).
