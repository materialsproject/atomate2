import json

from atomate2.common.schemas.elastic import ElasticDocument


def test_elastic_document(test_dir):
    schema_path = test_dir / "schemas" / "elastic.json"
    schema_ref = json.loads(schema_path.read_text())

    ElasticDocument(**schema_ref)
