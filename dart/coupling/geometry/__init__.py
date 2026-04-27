"""Geometry subpackage: G1-to-G3 lofting, OBJ conversion, and MTG export."""

from .g1_to_g3 import G3Mesh, loft_organs, render_views
from .cplantbox_adapter import extract_organs_for_lofter, extract_organs_from_plant
from .sheath_mesher import mesh_sheath
from .obj_dart_converter import convert_obj_to_dart, convert_mapping_json_groups
from .cplantbox_to_mtg import cplantbox_to_mtg, write_mtg_file, read_mtg_with_arrays
from .tassel_billboards import append_tassel_billboards, spikelet_billboards
