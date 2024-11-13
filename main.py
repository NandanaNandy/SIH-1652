"""Fastapi app"""

from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse

from v2 import FileResponse, process_file_400, uploadedFileDto

app = FastAPI(
    title="File Upload API",
    version="0.1.0",
    description="API to upload and process files",
)


@app.post(
    "/process-file",
    response_model=FileResponse,
    summary="Process the uploaded file",
    description="Process the uploaded image or pdf file",
    response_description="File processing response",
)
async def process_file(file: UploadFile = uploadedFileDto):
    """Extract and Verify the uploaded file"""

    if file.content_type not in ["image/jpeg", "image/png", "application/pdf"]:
        return process_file_400

    # Dummy value return for now
    return JSONResponse(
        content={
            "message": "File received",
            "filename": file.filename,
            "content_type": file.content_type,
        }
    )
