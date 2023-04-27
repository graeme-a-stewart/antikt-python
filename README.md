# AntiKt-Python

This package is an implementation of the [AntiKt algorithm](https://arxiv.org/abs/0802.1189) in Python, with both a basic version and a tiled implementation.

There is a pure Python implementation and one using `numpy` plus `numba`.

This code is used in the CHEP 2023 Paper, *Polyglot Jet Finding*.

## Copyright

All files are Copyright (C) CERN, 2023 and provided under the Apache-2 license.

## Acknowledgements

The antikt N^2 tiled algorithm is adapted from ClusterSequence_Tiled_N2.cc c++ code 
from the Fastjet (<https://fastjet.fr>,  hep-ph/0512210,  arXiv:1111.6097).

The reimplementation of the algorithm by Philippe Gras in Julia was extremely
useful when developing this Python version (<https://github.com/grasph/AntiKt.jl>).
