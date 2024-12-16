"""Models for the v2 API."""

from pydantic import BaseModel


class FileResponse(BaseModel):
    """Response model for the /process-file endpoint."""

    message: str
    filename: str
    content_type: str
