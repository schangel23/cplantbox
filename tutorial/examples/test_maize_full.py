"""Test full maize parameterization (roots + stems + leaves)
Created for CPlantBox learning - demonstrates full plant visualization
"""
import sys; sys.path.append("../.."); sys.path.append("../../src/")

import plantbox as pb
import plantbox.visualisation.vtk_plot as vp

# Create plant
plant = pb.Plant()

# Load FULL maize parameters (roots + stems + leaves)
path = "../../modelparameter/structural/plant/"
plant.readParameters(path + "maize.xml")

# Optional: modify simulation time (default is 56 days)
# seed = plant.getOrganRandomParameter(pb.seed)[0]
# seed.simulationTime = 30  # shorter simulation

# Initialize and simulate
plant.initialize()
simtime = 40  # days (adjust as needed)
print(f"Simulating maize for {simtime} days...")
plant.simulate(simtime, True)  # True = verbose

# Print some statistics
print("\n--- Plant Statistics ---")

# Get organ counts and lengths per type
roots = plant.getOrgans(pb.root)
stems = plant.getOrgans(pb.stem)
leaves = plant.getOrgans(pb.leaf)

# Calculate lengths by summing individual organ lengths
root_length = sum([r.getLength() for r in roots]) if roots else 0
stem_length = sum([s.getLength() for s in stems]) if stems else 0
leaf_length = sum([l.getLength() for l in leaves]) if leaves else 0

print(f"Number of roots: {len(roots)}, total length: {root_length:.1f} cm")
print(f"Number of stems: {len(stems)}, total length: {stem_length:.1f} cm")
print(f"Number of leaves: {len(leaves)}, total length: {leaf_length:.1f} cm")

# Export VTP for ParaView
plant.write("results/maize_full.vtp")
print("\nExported to results/maize_full.vtp")

# Visualize with VTK - color by organ type
# 2 = root (red), 3 = stem (green), 4 = leaf (blue)
print("\nOpening visualization (press 'q' to close)...")
vp.plot_plant(plant, "organType")
