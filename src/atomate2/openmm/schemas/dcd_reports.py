from pydantic import BaseModel, Field


class DCDReports(BaseModel):
    location: str = Field(None, description="Location of the DCD file")  # this should be a S3 location
    # TODO: Add host?
    report_interval: int = Field(None, description="Report interval")
    enforce_periodic_box: bool = Field(None, description="Wrap particles or not")

    @classmethod
    def from_dcd_file(cls, dcd_file):
        # TODO: will somehow need to interface with the additional store?
        return