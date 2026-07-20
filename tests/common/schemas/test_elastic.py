import json

import pytest
from monty.json import MontyEncoder

from atomate2.common.schemas.elastic import (
    DerivedProperties,
    ElasticDocument,
    ElasticTensorDocument,
    FittingData,
)


def test_elastic_document(test_dir):
    schema_path = test_dir / "schemas" / "elastic.json"
    schema_ref = json.loads(schema_path.read_text())

    doc = ElasticDocument(**schema_ref)
    validated = ElasticDocument.model_validate_json(json.dumps(doc, cls=MontyEncoder))
    assert isinstance(validated, ElasticDocument)


# schemas where all fields have default values
@pytest.mark.parametrize(
    "model_cls",
    [ElasticDocument, ElasticTensorDocument, DerivedProperties, FittingData],
)
def test_model_validate(model_cls):
    model_cls.model_validate_json(json.dumps(model_cls(), cls=MontyEncoder))


def test_estimate_expansion_order():
    """Order inference from strain-data structure, not calculation count."""
    import numpy as np
    from pymatgen.analysis.elasticity import Strain

    from atomate2.common.analysis.elastic import (
        estimate_expansion_order,
        get_default_strain_states,
    )

    def strains(states, magnitudes):
        return [Strain.from_voigt(m * np.array(s)) for s in states for m in magnitudes]

    states2 = get_default_strain_states(2)
    states3 = get_default_strain_states(3)

    # default order-2 sampling (6 states x 4 magnitudes = 24 strains)
    assert estimate_expansion_order(strains(states2, [-0.01, -0.005, 0.005, 0.01])) == 2

    # default order-3 sampling (14 states x 6 magnitudes = 84 strains)
    mags3 = [-0.01, -0.0066, -0.0033, 0.0033, 0.0066, 0.01]
    assert estimate_expansion_order(strains(states3, mags3)) == 3

    # regression for the old ``len(stresses) < 70`` heuristic: densely sampled
    # 2nd-order data (6 states x 12 magnitudes = 72 strains) must stay order 2
    dense = [m for m in np.linspace(-0.01, 0.01, 13) if abs(m) > 1e-10]
    assert len(strains(states2, dense)) == 72
    assert estimate_expansion_order(strains(states2, dense)) == 2

    # order-3 strain states but too few magnitudes per state -> order 2
    assert estimate_expansion_order(strains(states3, [-0.01, 0.01])) == 2

    # empty input falls back to order 2
    assert estimate_expansion_order([]) == 2
