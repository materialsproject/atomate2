"""Common utilities for atomate2."""

def get_transformations(
    transformations: tuple[str, ...], params: tuple[dict, ...] | None
):
    """Get instantiated transformation objects from their names and parameters."""
    params = ({},) * len(transformations) if params is None else params

    if len(params) != len(transformations):
        raise ValueError("Number of transformations and parameters must be the same.")

    transformation_objects = []
    for transformation, transformation_params in zip(transformations, params):
        found = False
        for m in (
            "advanced_transformations",
            "site_transformations",
            "standard_transformations",
        ):
            from importlib import import_module

            mod = import_module(f"pymatgen.transformations.{m}")

            try:
                t_cls = getattr(mod, transformation)
                found = True
                continue
            except AttributeError:
                pass

        if not found:
            raise ValueError(f"Could not find transformation: {transformation}")

        t_obj = t_cls(**transformation_params)
        transformation_objects.append(t_obj)
    return transformation_objects