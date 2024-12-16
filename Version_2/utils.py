"""utitlity functions for the API"""

from fastapi import File

uploadedFileDto = File(..., description="file to upload `IMAGE` or `PDF`", title="file")
