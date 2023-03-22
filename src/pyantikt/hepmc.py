import pyhepmc
import vector
import logging

import numpy as np

# This is not working as I want...
#logging.getLogger(__name__)#.addHandler(logging.NullHandler())
# logger.setLevel(logging.DEBUG)

def read_jet_particles(file="../data/events.hepmc3/events.hepmc3", skip=0, limit_particles=-1):
    """Read a jet from a HepMC3 input file, returning a normal Python list of particle vectors"""
    with pyhepmc.open(file) as jets:
        jet = jets.read()
        for _ in range(skip):
            jet = jets.read()

    particles = []
    for particle in jet.particles:
        # Particle status 1 = Undecayed physical particle
        if particle.status == 1:
            particles.append(vector.obj(px=particle.momentum.px,
                                        py=particle.momentum.py,
                                        pz=particle.momentum.pz,
                                        e=particle.momentum.e))
            # Artificially limit particles for debugging
            if len(particles) == limit_particles:
                break

    return particles
