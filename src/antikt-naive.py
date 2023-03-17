#! /usr/bin/env python3
"""Anti-Kt jet finder, naive unoptimised implementation"""

import argparse
import numpy as np
import pyhepmc
import vector

from copy import deepcopy


def read_jet_particles(file="../data/events.hepmc3/events.hepmc3", skip=0):
    """Read a jet from a HepMC3 input file"""
    with pyhepmc.open(file) as jets:
        jet = jets.read()
        for _ in range(skip):
            jet = jets.read()

    # Count final state particles here, so arrays can be setup with the correct size
    fsparticles = []
    for iparticle, particle in enumerate(jet.particles):
        # Particle status 1 = Undecayed physical particle
        if particle.status == 1:
            fsparticles.append(iparticle)
    print(f"Final state particles: {len(fsparticles)}")
    nparticles = len(fsparticles)

    # Pick only final state particles and construct arrays of properties
    px = np.zeros(nparticles)
    py = np.zeros(nparticles)
    pz = np.zeros(nparticles)
    E = np.zeros(nparticles)
    for iparticle, fsparticle in enumerate(fsparticles):
        px[iparticle] = jet.particles[fsparticle].momentum.px
        py[iparticle] = jet.particles[fsparticle].momentum.py
        pz[iparticle] = jet.particles[fsparticle].momentum.pz
        E[iparticle] = jet.particles[fsparticle].momentum.e

    particles = vector.array({"px": px, "py": py, "pz": pz, "e": E})

    return particles


def akt_distance(pj1, pj2, r2=0.16):
    """Return the antikt metric between two pseudojets (note the ^-2 in the momentum term)
    If the two pseudojets are the same one, we return the beamline distance"""

    # Beamline distance
    if pj1 == pj2:
        return pj1.pt**-2
    # AntiKt distance
    delta2_distance = (pj1.y - pj2.y) ** 2 + (pj1.phi - pj2.phi) ** 2
    min_momentum = min(pj1.pt**-2, pj2.pt**-2)
    return min_momentum * delta2_distance / r2


def calculate_antikt_distance(particles, r2=0.16):
    """Calculate the antikt distances between particles, using an upper triangular array"""
    distance_array = np.full((len(particles), len(particles)), 66666.6, dtype=float)

    # This is a loop, could be better implemented?
    for i in range(len(particles)):
        for j in range(i, len(particles)):
            distance_array[i][j] = akt_distance(particles[i], particles[j], r2)

    return distance_array


def find_minumum_distance_index(distance_array):
    min_value = np.min(distance_array)
    min_index = np.argmin(distance_array, keepdims=True)
    min_indexes = np.unravel_index(min_index, np.shape(distance_array))
    min_i = min_indexes[0][0][0]
    min_j = min_indexes[1][0][0]

    print(f"Minumum value is {min_value} at [{min_i}, {min_j}]")

    return (min_i, min_j)


def merge_pseudo_jets(
    current_particles, current_distances, jet_merge_i, jet_merge_j, r2=0.16
):
    """Merge two pseudo jets and return the updated list of particles and distances"""

    # First calculate the new merged pseudoJet
    if jet_merge_i != jet_merge_j:
        merged_jet = current_particles[jet_merge_i] + current_particles[jet_merge_j]

    # The idea here is to slice out jets i and j from the existing matrix
    # N.B. np.insert, np.delete and np.concatenate are not supported for numpy arrays of vectors!
    particle_mask = np.arange(len(current_particles))
    particle_mask = np.delete(particle_mask, (jet_merge_i, jet_merge_j))

    # Can't seem to append to a numpy array of vectors, so need to recreate by pieces (!)
    new_px = current_particles[particle_mask].px
    new_py = current_particles[particle_mask].py
    new_pz = current_particles[particle_mask].pz
    new_e = current_particles[particle_mask].E

    if jet_merge_i != jet_merge_j:
        new_px = np.insert(new_px, len(current_particles[particle_mask]), merged_jet.px)
        new_py = np.insert(new_py, len(current_particles[particle_mask]), merged_jet.py)
        new_pz = np.insert(new_pz, len(current_particles[particle_mask]), merged_jet.pz)
        new_e = np.insert(new_e, len(current_particles[particle_mask]), merged_jet.E)

    next_particles = vector.array(
        {"px": new_px, "py": new_py, "pz": new_pz, "e": new_e}
    )
    # print(type(current_particles), type(next_particles))
    # print(len(current_particles), len(next_particles))
    # next_particles = np.((current_particles[particle_mask], [merged_jet]))
    # Then append the new merged jet
    # next_particles = np.insert(next_particles, len(next_particles), merged_jet)

    # Recaculate the distances (TODO: update to slicing of the current array,
    # which should work as the distance array is just floats)
    next_distances = calculate_antikt_distance(next_particles, r2=r2)

    return next_particles, next_distances


def antikt_jet_finder(initial_particles, r=0.4):
    """Run the anti-kt jet algorithm over a set of particles, returning
    an array with the final jet constituents"""

    r2 = r * r
    current_particles = deepcopy(initial_particles)
    current_distances = calculate_antikt_distance(current_particles, r2=r2)
    print(current_distances)
    final_jets = []

    # Iterate until done
    while len(current_particles) > 0:
        (jet_merge_i, jet_merge_j) = find_minumum_distance_index(current_distances)

        if jet_merge_i == jet_merge_j:
            # We have a final jet
            print(f"Found a final jet, index {jet_merge_i}")
            final_jets.append(current_particles[jet_merge_i])
            next_particles, next_distances = merge_pseudo_jets(
                current_particles, current_distances, jet_merge_i, jet_merge_j
            )

        else:
            # We merge i and j
            print(f"Will now merge jet indexes {jet_merge_i} and {jet_merge_j}")
            next_particles, next_distances = merge_pseudo_jets(
                current_particles, current_distances, jet_merge_i, jet_merge_j, r2=r2
            )

        print(f"Pseudo jet array is now {len(next_particles)} long")
        current_particles = next_particles
        current_distances = next_distances

    print(f"Found {len(final_jets)} jets")
    for njet, jet in enumerate(final_jets):
        print(f"Jet {njet}: {jet}")


def main():
    initial_particles = read_jet_particles()
    print(initial_particles[0])

    antikt_jet = antikt_jet_finder(initial_particles)


if __name__ == "__main__":
    main()
