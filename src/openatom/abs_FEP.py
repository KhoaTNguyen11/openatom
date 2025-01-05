from typing import List, Tuple, Optional
from numpy import ndarray
import openmm
import openmm.app
from openmm import unit
import xml.etree.ElementTree as ET
import copy
import numpy as np
from collections import defaultdict
from openatom.merge_systems import *

def _make_ligand_vdw_forces(lig: ET.Element) -> ET.Element:
    '''
    Make the custom bond forces for the ligand, consider adding virtual bond link to the ligand and handle
    the LJ potential inside the ligand
    Args:
    lig (xml.etree.ElementTree.Element): ligand element
    Returns:
    xml.etree.ElementTree.Element: the custom bond forces
    '''
    formula = [
        "4*eps*((sig/r)^12 - (sig/r)^6)",
    ]
    force = ET.Element(
        "Force",
        {
            "energy": ";".join(formula),
            "name": "CustomBondForce",
            "type": "CustomBondForce",
            "version": "3",
            "usesPeriodic": "0",
        },
    )

    ET.SubElement(force, "GlobalParameters")
    perparticle_parameters = ET.SubElement(force, "PerBondParameters")
    perparticle_parameters.append(ET.Element("Parameter", {"name": "eps"}))
    perparticle_parameters.append(ET.Element("Parameter", {"name": "sig"}))

    ET.SubElement(force, "ComputedValues")
    ET.SubElement(force, "EnergyParameterDerivatives")
    ET.SubElement(force, "Functions")

    # Initialize the parameters for the particles
    particle_params = {}

    lig_nf = None
    for f in lig.iterfind("./Forces/Force"):
        if f.get("type") == "NonbondedForce":
            lig_nf = f
            break

    for i, p in enumerate(lig_nf.iterfind("./Particles/Particle")):
        j = int(lig.find("./Particles")[i].get("idx"))
        eps = float(p.get("eps"))
        sig = float(p.get("sig"))
        particle_params[j] = {
            "eps": eps,
            "sig": sig,
        }

    # Update parameters from NonbondedForce
    lig_nf = None
    for f in lig.iterfind("./Forces/Force"):
        if f.get("type") == "NonbondedForce":
            lig_nf = f
            break

    bonds = ET.SubElement(force, "Bonds")

    for e in lig_nf.iterfind("./Exceptions/Exception"):
        i1, i2 = int(e.get("p1")), int(e.get("p2"))
        j1, j2 = _get_idx(lig, [i1, i2])
        j1, j2 = sorted([j1, j2])
        eps = float(e.get("eps"))
        sig = float(e.get("sig"))
        ET.SubElement(
            bonds,
            "Bond",
            {
                "p1": str(j1),
                "p2": str(j2),
                "param1": str(eps),
                "param2": str(sig),
            },
        )

    # then add the virtual bond link each atom in the ligand if they are not already in our bond list
    for i, p in enumerate(lig_nf.iterfind("./Particles/Particle")):
        j = int(lig.find("./Particles")[i].get("idx"))
        for k in range(i + 1, len(lig_nf.findall("./Particles/Particle"))):
            l = int(lig.find("./Particles")[k].get("idx"))
            if (j, l) not in [(int(b.get("p1")), int(b.get("p2"))) for b in bonds.findall("./Bond")]:
                eps = (particle_params[j]["eps"] *
                       particle_params[l]["eps"]) ** 0.5
                sig = 0.5 * (particle_params[j]
                             ["sig"] + particle_params[l]["sig"])
                ET.SubElement(
                    bonds,
                    "Bond",
                    {
                        "p1": str(j),
                        "p2": str(l),
                        "param1": str(eps),
                        "param2": str(sig),
                    },
                )

    return force


def _make_ligand_coul_forces(lig: ET.Element, lambdas: Tuple[float, float]) -> ET.Element:
    '''
    add back the coulombic forces to the ligand
    Args:
    lig (xml.etree.ElementTree.Element): ligand element
    lambdas (tuple[float, float]): lambda values, we add back 1-lambda[0][0] to the charge of the ligand because we scaled the charge in nonbonded forces
    Returns:
    xml.etree.ElementTree.Element: the custom bond forces to show the coulombic forces
    '''
    formula = [
        "ke * q / r",
    ]
    force = ET.Element(
        "Force",
        {
            "energy": ";".join(formula),
            "name": "CustomBondForce",
            "type": "CustomBondForce",
            "version": "3",
            "usesPeriodic": "0",
        },
    )

    eps_0 = 8.854187817e-12 * unit.farad / unit.meter
    # 138.935485 kJ/mol in scale length of nm and charge of e
    ke = (1.602176634e-19 ** 2 * 6.0221e23) / (4 * np.pi * eps_0 * 1e-9 * 1000)

    global_parameters = ET.SubElement(force, "GlobalParameters")
    ET.SubElement(global_parameters, "Parameter", {
                  "name": "ke", "default": f'{ke._value}'})
    perparticle_parameters = ET.SubElement(force, "PerBondParameters")
    perparticle_parameters.append(ET.Element("Parameter", {"name": "q"}))

    ET.SubElement(force, "ComputedValues")
    ET.SubElement(force, "EnergyParameterDerivatives")
    ET.SubElement(force, "Functions")

    lambda_coul = lambdas[0][0]

    # Initialize the parameters for the particles
    particle_params = {}

    # Update parameters from NonbondedForce
    lig_nf = None
    for f in lig.iterfind("./Forces/Force"):
        if f.get("type") == "NonbondedForce":
            lig_nf = f
            break

    for i, p in enumerate(lig_nf.iterfind("./Particles/Particle")):
        j = int(lig.find("./Particles")[i].get("idx"))
        q = float(p.get("q"))
        particle_params[j] = {
            "charge": q,
        }

    bonds = ET.SubElement(force, "Bonds")

    # check exceptions
    for e in lig_nf.iterfind("./Exceptions/Exception"):
        i1, i2 = int(e.get("p1")), int(e.get("p2"))
        j1, j2 = _get_idx(lig, [i1, i2])
        j1, j2 = sorted([j1, j2])
        q = float(e.get("q")) * (1 - lambda_coul)
        ET.SubElement(
            bonds,
            "Bond",
            {
                "p1": str(j1),
                "p2": str(j2),
                "param1": str(q),
            },
        )

    # then add the virtual bond link each atom in the ligand if they are not already in our bond list
    for i, p in enumerate(lig_nf.iterfind("./Particles/Particle")):
        j = int(lig.find("./Particles")[i].get("idx"))
        for k in range(i + 1, len(lig_nf.findall("./Particles/Particle"))):
            l = int(lig.find("./Particles")[k].get("idx"))
            if (j, l) not in [(int(b.get("p1")), int(b.get("p2"))) for b in bonds.findall("./Bond")]:
                q = particle_params[j]["charge"] * \
                    particle_params[l]["charge"] * (1-lambda_coul ** 2)
                ET.SubElement(
                    bonds,
                    "Bond",
                    {
                        "p1": str(j),
                        "p2": str(l),
                        "param1": str(q),
                    },
                )

    return force


def _make_custom_forces(lig: ET.Element, lambdas: Tuple[float, float], env: ET.Element = None) -> ET.Element:
    """
    Make the custom soft core forces for the ligand and environment

    Args:
    lig (xml.etree.ElementTree.Element): ligand element
    lambdas (tuple[float, float]): lambda values
    env (xml.etree.ElementTree.Element): environment element

    Returns:
    xml.etree.ElementTree.Element: the custom forces
    """
    formula = [
        "4*epsilon*lambda*(1/(alpha*(1-lambda) + (r/sigma)^6)^2 - 1/(alpha*(1-lambda) + (r/sigma)^6))",
        "epsilon = sqrt(eps1*eps2)",
        "sigma = 0.5*(sig1+sig2)",
        "alpha = 0.5",
    ]
    lig_nf = None
    for f in lig.iterfind("./Forces/Force"):
        if f.get("type") == "NonbondedForce":
            lig_nf = f

    env_nf = None
    if env is not None:
        method = '2'  # 2: CutoffPeriodic
        for f in env.iterfind("./Forces/Force"):
            if f.get("type") == "NonbondedForce":
                env_nf = f

        cut_off = env_nf.get("cutoff")
        switch_distance = env_nf.get("switchingDistance")
        long_range_correction = env_nf.get("dispersionCorrection")
    else:
        method = '1'  # CutOffNonPeriodic
        cut_off = lig_nf.get("cutoff")
        switch_distance = lig_nf.get("switchingDistance")
        long_range_correction = lig_nf.get("dispersionCorrection")

    force = ET.Element(
        "Force",
        {
            "cutoff": cut_off,
            "energy": ";".join(formula),
            "forceGroup": "0",
            "method": method,
            "name": "CustomNonbondedForce",
            "switchingDistance": switch_distance,
            "type": "CustomNonbondedForce",
            "useLongRangeCorrection": long_range_correction,
            "useSwitchingFunction": "1",
            "version": "3",
        },
    )

    perparticle_parameters = ET.SubElement(force, "PerParticleParameters")
    perparticle_parameters.append(ET.Element("Parameter", {"name": "eps"}))
    perparticle_parameters.append(ET.Element("Parameter", {"name": "sig"}))

    global_parameters = ET.SubElement(force, "GlobalParameters")
    lambda_vdw = lambdas[0][1]
    ET.SubElement(global_parameters, "Parameter", {
                  "name": "lambda", "default": str(lambda_vdw)})

    ET.SubElement(force, "ComputedValues")
    ET.SubElement(force, "EnergyParameterDerivatives")
    ET.SubElement(force, "Functions")

    particles = ET.SubElement(force, "Particles")

    for p in lig.iterfind("./Particles/Particle"):
        ET.SubElement(particles, "Particle", {"param1": "0", "param2": "0"})

    if env is not None:
        for p in env.iterfind("./Particles/Particle"):
            ET.SubElement(particles, "Particle", {
                          "param1": "0", "param2": "0"})

    for i, p in enumerate(lig_nf.iterfind("./Particles/Particle")):
        j = int(lig.find("./Particles")[i].get("idx"))
        eps = float(p.get("eps"))
        sig = float(p.get("sig"))
        particles[j].set("param1", str(eps))
        particles[j].set("param2", str(sig))

    if env_nf is not None:
        for i, p in enumerate(env_nf.iterfind("./Particles/Particle")):
            j = int(env.find("./Particles")[i].get("idx"))
            particles[j].set("param1", p.get("eps"))
            particles[j].set("param2", p.get("sig"))

    interaction_groups = ET.SubElement(force, "InteractionGroups")
    group = ET.SubElement(interaction_groups, "InteractionGroup")
    set1 = ET.SubElement(group, "Set1")
    set2 = ET.SubElement(group, "Set2")

    for i, p in enumerate(lig.iterfind("./Particles/Particle")):
        j = p.get("idx")
        ET.SubElement(set1, "Particle", {"index": j})

    if env is not None:
        for i, p in enumerate(env.iterfind("./Particles/Particle")):
            j = p.get("idx")
            ET.SubElement(set2, "Particle", {"index": j})

    exclusions = ET.SubElement(force, "Exclusions")
    exclusion_set = set()

    for f in lig.iterfind("./Forces/Force"):
        if f.get("type") == "NonbondedForce":
            for e in f.iterfind("./Exceptions/Exception"):
                i1, i2 = int(e.get("p1")), int(e.get("p2"))
                j1, j2 = sorted(_get_idx(lig, [i1, i2]))
                if (j1, j2) not in exclusion_set:
                    ET.SubElement(exclusions, "Exclusion", {
                                  "p1": str(j1), "p2": str(j2)})
                    exclusion_set.add((j1, j2))

    if env is not None:
        for e in env_nf.iterfind("./Exceptions/Exception"):
            i1, i2 = int(e.get("p1")), int(e.get("p2"))
            j1, j2 = sorted(_get_idx(env, [i1, i2]))
            if (j1, j2) not in exclusion_set:
                ET.SubElement(exclusions, "Exclusion", {
                              "p1": str(j1), "p2": str(j2)})
                exclusion_set.add((j1, j2))

    return force



def make_abs_alchemy_system(
    lig: ET.Element,
    lig_top: openmm.app.Topology,
    lig_coords: ndarray,
    lambdas: List[Tuple[float, float]],
    environment: ET.Element,
    env_top: openmm.app.Topology,
    env_coords: ndarray,
) -> Tuple[ET.Element, ndarray]:
    '''make the alchemical system for absolute hydration free energy calculation'''

    if environment is not None:
        system = ET.Element("System", environment.attrib)
        system.append(environment.find("./PeriodicBoxVectors"))
    else:
        system = ET.Element("System", lig.attrib)
        system.append(lig.find("./PeriodicBoxVectors"))

    particles, topology = _merge_particles_and_topology(
        lig, lig_top,  environment, env_top
    )

    system.append(particles)

    constraints = _merge_constraints(lig, environment)
    system.append(constraints)

    forces = _merge_forces(lig, lambdas, environment)
    system.append(forces)

    n = len(system.findall("./Particles/Particle"))
    coor = np.zeros((n, 3))
    for i, p in enumerate(lig.findall("./Particles/Particle")):

        j = int(p.get("idx"))
        coor[j] = lig_coords[i]

    if environment is not None:
        for i, p in enumerate(environment.findall("./Particles/Particle")):
            j = int(p.get("idx"))
            coor[j] = env_coords[i]

    coor = np.array(coor)

    return system, topology, coor


