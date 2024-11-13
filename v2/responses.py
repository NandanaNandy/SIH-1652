"""Custom responses for the API."""
from fastapi.responses import JSONResponse

process_file_400 = JSONResponse(
    status_code=400,
    content={"message": "Invalid file type. Please upload an image or pdf."},
)
