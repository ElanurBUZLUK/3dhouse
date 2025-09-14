"""
Geometry module for Sketch2House3D.
Contains 3D geometry generation and manipulation.
"""

from .walls_builder import WallsBuilder, build_walls
from .openings_boolean import OpeningsBoolean, OpeningMeshGenerator, create_openings
from .roof_builder import (
    GableRoofBuilder, build_gable_roof,
    HipRoofBuilder, build_hip_roof,
    ShedRoofBuilder, build_shed_roof,
    FlatRoofBuilder, build_flat_roof
)

__all__ = [
    'WallsBuilder',
    'build_walls',
    'OpeningsBoolean',
    'OpeningMeshGenerator',
    'create_openings',
    'GableRoofBuilder',
    'build_gable_roof',
    'HipRoofBuilder',
    'build_hip_roof',
    'ShedRoofBuilder',
    'build_shed_roof',
    'FlatRoofBuilder',
    'build_flat_roof'
]
