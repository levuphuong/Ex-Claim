from fastapi import FastAPI
from fastapi.responses import FileResponse
import shutil
import os

app = FastAPI()

@app.get("/download")
def download_outputs():
    if not os.path.exists("outputs.zip"):
        shutil.make_archive("outputs", "zip", "outputs")
    return FileResponse(
        "outputs.zip",
        filename="outputs.zip",
        media_type="application/octet-stream"
    )
