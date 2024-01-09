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
