import json

try:
    from openff.interchange import Interchange
except ImportError:

    class Interchange:  # type: ignore[no-redef]
        """Dummy class for failed imports of Interchange."""

        def model_validate(self, _: str) -> None:
            """Parse raw is the first method called on the Interchange object."""
            raise ImportError(
                "openff-interchange must be installed for OpenMM makers to"
                "to support OpenFF Interchange objects."
            )

class LammpsInterchange(Interchange):
    """
    A subclass of Interchange that adds a method to convert the object to a dictionary.
    """
    
    @classmethod
    def from_interchange(cls, interchange: Interchange) -> None:
        """
        Load an Interchange object.
        """
        
        return cls(topology=interchange.topology,
                  collections=interchange.collections,
                  box=interchange.box,
                  positions=interchange.positions,
                  velocities=interchange.velocities,
                  mdconfig=interchange.mdconfig
                  )
    
    def as_dict(self) -> dict:
        """
        Convert the Interchange object to a dictionary.
        """
        return json.loads(self.json())
    
    @classmethod
    def from_dict(cls, d: dict) -> "LammpsInterchange":
        """
        Load an Interchange object from a dictionary.
        """
        return cls(**d)