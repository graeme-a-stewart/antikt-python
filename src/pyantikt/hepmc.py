import pyhepmc
from pyantikt.pseudojet import PseudoJet

import numpy as np


def read_jet_particles(
    file="../data/events.hepmc3/events.hepmc3", skip=0, nevents=1, limit_particles=-1
):
    """Read a jet from a HepMC3 input file, returning a normal Python list of particle vectors"""
    events = []
    with pyhepmc.open(file) as jets:
        for _ in range(skip):
            jet = jets.read()
        for jet in range(nevents):
            jet = jets.read()

            particles = []
            for particle in jet.particles:
                # Particle status 1 = Undecayed physical particle
                if particle.status == 1:
                    particles.append(
                        PseudoJet(
                            px=particle.momentum.px,
                            py=particle.momentum.py,
                            pz=particle.momentum.pz,
                            E=particle.momentum.e,
                        )
                    )
                    # Artificially limit particles for debugging
                    if len(particles) == limit_particles:
                        break
            events.append(particles)
    return events
