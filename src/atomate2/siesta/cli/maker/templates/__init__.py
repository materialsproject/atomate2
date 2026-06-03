"""Workflow templates for atomate2siesta-maker CLI."""

from atomate2.siesta.cli.maker.templates.base import WorkflowTemplate
from atomate2.siesta.cli.maker.templates.basic import (
    BandsTemplate,
    DOSTemplate,
    OpticalTemplate,
    PDOSTemplate,
    RelaxTemplate,
    StaticTemplate,
)
from atomate2.siesta.cli.maker.templates.mechanical import (
    BulkModulusTemplate,
    ElasticTemplate,
    EOSTemplate,
)
from atomate2.siesta.cli.maker.templates.surface import (
    AdsorptionOptimizationTemplate,
    AdsorptionTemplate,
    MultiSurfaceTemplate,
    SurfaceTemplate,
)
from atomate2.siesta.cli.maker.templates.transition import NebTemplate
from atomate2.siesta.cli.maker.templates.vibrational import (
    GruneisenTemplate,
    PhononTemplate,
    QHATemplate,
)

__all__ = [
    "WorkflowTemplate",
    "RelaxTemplate",
    "StaticTemplate",
    "BandsTemplate",
    "DOSTemplate",
    "PDOSTemplate",
    "OpticalTemplate",
    "PhononTemplate",
    "GruneisenTemplate",
    "QHATemplate",
    "EOSTemplate",
    "ElasticTemplate",
    "BulkModulusTemplate",
    "NebTemplate",
    "SurfaceTemplate",
    "AdsorptionTemplate",
    "MultiSurfaceTemplate",
    "AdsorptionOptimizationTemplate",
]

# Template registry
TEMPLATES = {
    # Basic
    "relax": RelaxTemplate(),
    "static": StaticTemplate(),
    "bands": BandsTemplate(),
    "dos": DOSTemplate(),
    "pdos": PDOSTemplate(),
    "optical": OpticalTemplate(),
    # Vibrational
    "phonon": PhononTemplate(),
    "gruneisen": GruneisenTemplate(),
    "qha": QHATemplate(),
    # Mechanical
    "eos": EOSTemplate(),
    "elastic": ElasticTemplate(),
    "bulk-modulus": BulkModulusTemplate(),
    # Transition states
    "neb": NebTemplate(),
    # Surface/Catalysis
    "surface": SurfaceTemplate(),
    "adsorption": AdsorptionTemplate(),
    "multi-surface": MultiSurfaceTemplate(),
    "adsorption-optimize": AdsorptionOptimizationTemplate(),
}
