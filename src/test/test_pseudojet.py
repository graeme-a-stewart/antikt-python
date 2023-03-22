#! /usr/bin/env python3

# TODO - make me a proper unittest...!

import vector
from pyantikt.pseudojet import pseudojet

px=0.22337332139323995
py=0.013200084794337135
pz=-0.18261482717845642
E=0.28882184483560597

vpj = vector.obj(px=px, py=py, pz=pz, E=E)

pjpj = pseudojet(px=px, py=py, pz=pz, E=E)

print(vpj.rapidity, pjpj.rap)
print(vpj.phi, pjpj.phi)
print(vpj.pt2, pjpj.pt2)
