"""Geometry subpackage: G1-to-G3 lofting and OBJ conversion."""

from .g1_to_g3 import G3Mesh, loft_organs, render_views
from .cplantbox_adapter import extract_organs_for_lofter, extract_organs_from_plant
from .obj_dart_converter import convert_obj_to_dart, convert_mapping_json_groups
