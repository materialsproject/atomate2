from atomate2.vasp.sets.base import VaspInputSetGenerator


class ElasticDeformationSetGenerator(VaspInputSetGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._config_dict["KPOINTS"] = {"grid_density": 7000}

    def get_incar_updates(
        self, structure, prev_incar=None, bandgap=0, vasprun=None, outcar=None
    ) -> dict:
        return {
            "IBRION": 2,
            "ISIF": 2,
            "ENCUT": 700,
            "EDIFF": 1e-7,
            "LAECHG": False,
            "EDIFFG": -0.001,
            "LREAL": False,
            "ALGO": "Normal",
            "NSW": 99,
            "LCHARG": False,
        }
