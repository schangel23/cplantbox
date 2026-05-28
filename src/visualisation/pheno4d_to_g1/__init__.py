"""Pheno4D point cloud to CPlantBox G1 pipeline.

Converts organ-labeled LiDAR point clouds (Pheno4D format) into
CPlantBox-compatible MappedSegments representations.

Usage:
    from plantbox.visualisation.pheno4d_to_g1 import pheno4d_to_cplantbox, pheno4d_to_dart

    ms = pheno4d_to_cplantbox('Maize01/M01_0325_a.txt')
    pheno4d_to_dart('Maize01/M01_0325_a.txt', 'output/maize01_dart')
"""

from .pipeline import pheno4d_to_cplantbox, pheno4d_to_dart, process_time_series
from .loader import load_pheno4d
from .segmenter import segment_maize
from .skeletonizer import skeletonize_stem, skeletonize_leaf, resample_path
from .g1_builder import build_mapped_segments
from .validate import validate_g1, visualize_skeleton
from .graph_refinement import (
    build_skeleton_graph,
    remove_ground_nodes,
    split_overlapping_leaves,
    identify_stem_and_leaves,
    prune_spurious_branches,
    segment_point_cloud,
    export_graph_diagnostics,
)
