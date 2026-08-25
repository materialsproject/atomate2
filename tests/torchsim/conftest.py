import importlib.metadata


def _is_backend_missing(package_names: str | list[str]) -> bool:
    if isinstance(package_names, str):
        package_names = [package_names]

    missing = False
    try:
        for pkg_name in package_names:
            importlib.metadata.distribution(pkg_name)
    except importlib.metadata.PackageNotFoundError:
        missing = True

    return missing


_SKIP_FAIRCHEM = _is_backend_missing("fairchem-core")
_SKIP_MACE = _is_backend_missing("mace-torch")
_SKIP_MATTERSIM = _is_backend_missing("mattersim")
_SKIP_METATOMIC = _is_backend_missing(["metatomic_torchsim", "upet"])
_SKIP_NEQUIP = _is_backend_missing("nequip")
_SKIP_ORB = _is_backend_missing("orb_models")
_SKIP_SEVENNET = _is_backend_missing("sevenn")
