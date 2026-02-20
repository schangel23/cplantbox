"""Visualize the multiple plants simulation results"""

import plantbox.visualisation.vtk_plot as vp
import plantbox.visualisation.vtk_tools as vt

# Load the combined VTP file
pd = vt.read_vtp("results/multplantsys_all.vtp")

# Create visualization
vp.plot_roots(pd, "subType", "Multiple Plants (3x3 Grid)", render=True, interactiveImage=True)
