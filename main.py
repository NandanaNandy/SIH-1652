from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()


@app.post("/process-file/")
async def process_file(file: UploadFile = File(...)):
    """Extract and Verify the uploaded file"""

    if file.content_type not in ["image/jpeg", "image/png", "application/pdf"]:
        return JSONResponse(
            status_code=400,
            content={"message": "Invalid file type. Please upload an image or pdf."},
        )

    # Dummy value return for now
    return JSONResponse(
        content={
            "message": "File received",
            "filename": file.filename,
            "content_type": file.content_type,
        }
    )
