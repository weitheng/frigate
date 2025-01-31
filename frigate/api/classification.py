"""Object classification APIs."""

import logging
import os
import random
import shutil
import string

from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import JSONResponse
from pathvalidate import sanitize_filename

from frigate.api.defs.tags import Tags
from frigate.const import FACE_DIR
from frigate.embeddings import EmbeddingsContext

logger = logging.getLogger(__name__)

router = APIRouter(tags=[Tags.events])


@router.get("/faces")
def get_faces():
    face_dict: dict[str, list[str]] = {}

    for name in os.listdir(FACE_DIR):
        face_dir = os.path.join(FACE_DIR, name)

        if not os.path.isdir(face_dir):
            continue

        face_dict[name] = []

        for file in sorted(
            os.listdir(face_dir),
            key=lambda f: os.path.getctime(os.path.join(face_dir, f)),
            reverse=True,
        ):
            face_dict[name].append(file)

    return JSONResponse(status_code=200, content=face_dict)


@router.post("/faces/{name}")
async def register_face(request: Request, name: str, file: UploadFile):
    try:
        if not request.app.frigate_config.face_recognition.enabled:
            return JSONResponse(
                status_code=400,
                content={"message": "Face recognition is not enabled.", "success": False},
            )

        if not name or not name.strip():
            return JSONResponse(
                status_code=400,
                content={"message": "Face name is required", "success": False},
            )

        context: EmbeddingsContext = request.app.embeddings
        file_content = await file.read()
        
        if not file_content:
            return JSONResponse(
                status_code=400,
                content={"message": "Empty file uploaded", "success": False},
            )

        result = context.register_face(name, file_content)
        return JSONResponse(
            status_code=200 if result.get("success", True) else 400,
            content=result,
        )
    except Exception as e:
        logger.error(f"Failed to register face: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": f"Failed to register face: {str(e)}", "success": False},
        )


@router.post("/faces/train/{name}/classify")
def train_face(request: Request, name: str, body: dict = None):
    try:
        if not request.app.frigate_config.face_recognition.enabled:
            return JSONResponse(
                status_code=400,
                content={"message": "Face recognition is not enabled.", "success": False},
            )

        if not name or not name.strip():
            return JSONResponse(
                status_code=400,
                content={"message": "Face name is required", "success": False},
            )

        json: dict[str, any] = body or {}
        training_file = os.path.join(
            FACE_DIR, f"train/{sanitize_filename(json.get('training_file', ''))}"
        )

        if not training_file or not os.path.isfile(training_file):
            return JSONResponse(
                content=(
                    {
                        "success": False,
                        "message": f"Invalid filename or no file exists: {training_file}",
                    }
                ),
                status_code=404,
            )

        rand_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
        new_name = f"{name}-{rand_id}.webp"
        new_file = os.path.join(FACE_DIR, f"{name}/{new_name}")
        
        os.makedirs(os.path.dirname(new_file), exist_ok=True)
        shutil.move(training_file, new_file)

        context: EmbeddingsContext = request.app.embeddings
        context.clear_face_classifier()

        return JSONResponse(
            content=(
                {
                    "success": True,
                    "message": f"Successfully saved {training_file} as {new_name}.",
                }
            ),
            status_code=200,
        )
    except Exception as e:
        logger.error(f"Failed to train face: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": f"Failed to train face: {str(e)}", "success": False},
        )


@router.post("/faces/reprocess")
def reclassify_face(request: Request, body: dict = None):
    if not request.app.frigate_config.face_recognition.enabled:
        return JSONResponse(
            status_code=400,
            content={"message": "Face recognition is not enabled.", "success": False},
        )

    json: dict[str, any] = body or {}
    training_file = sanitize_filename(json.get('training_file', ''))
    face_name = json.get('face_name')

    if not training_file:
        return JSONResponse(
            content={"success": False, "message": "Training file is required"},
            status_code=400,
        )

    # Check if file exists in train directory
    training_path = os.path.join(FACE_DIR, "train", training_file)
    if not os.path.isfile(training_path):
        # Check if file exists in face directory
        face_path = os.path.join(FACE_DIR, face_name, training_file)
        if not os.path.isfile(face_path):
            return JSONResponse(
                content={
                    "success": False,
                    "message": f"File not found: {training_file}"
                },
                status_code=404,
            )
        training_path = face_path

    try:
        context: EmbeddingsContext = request.app.embeddings
        
        # For files in face directories, move to train first
        if not training_path.startswith(os.path.join(FACE_DIR, "train")):
            train_path = os.path.join(FACE_DIR, "train", training_file)
            shutil.copy2(training_path, train_path)
            os.remove(training_path)
            training_path = train_path

        response = context.reprocess_face(training_path, face_name)

        return JSONResponse(
            content=response,
            status_code=200 if response.get('success', False) else 422,
        )
    except Exception as e:
        logger.error(f"Failed to reprocess face: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Failed to reprocess face: {str(e)}"},
        )


@router.post("/faces/{name}/delete")
def deregister_faces(request: Request, name: str, body: dict = None):
    context: EmbeddingsContext = request.app.embeddings
    
    if not request.app.frigate_config.face_recognition.enabled:
        return JSONResponse(
            status_code=400,
            content={"message": "Face recognition is not enabled.", "success": False},
        )

    json: dict[str, any] = body or {}
    list_of_ids = json.get("ids", [])
    delete_directory = json.get("delete_directory", False)

    if not list_of_ids:
        return JSONResponse(
            content={"success": False, "message": "Not a valid list of ids"},
            status_code=404,
        )

    face_dir = os.path.join(FACE_DIR, name)
    
    if not os.path.exists(face_dir):
        return JSONResponse(
            status_code=404,
            content={"message": f"Face '{name}' not found", "success": False},
        )

    try:
        if delete_directory:
            shutil.rmtree(face_dir)
        else:
            context.delete_face_ids(
                name, map(lambda file: sanitize_filename(file), list_of_ids)
            )
        
        context.clear_face_classifier()

        return JSONResponse(
            content={"success": True, "message": "Successfully deleted faces."},
            status_code=200,
        )
    except Exception as e:
        logger.error(f"Failed to delete face: {str(e)}")
        return JSONResponse(
            content={"success": False, "message": f"Failed to delete face: {str(e)}"},
            status_code=500,
        )


@router.post("/faces/{name}/create")
def create_face(name: str):
    """Create a new face directory without requiring an image."""
    folder = os.path.join(FACE_DIR, name)
    if os.path.exists(folder):
        return JSONResponse(
            status_code=400,
            content={"message": f"Face '{name}' already exists", "success": False},
        )
    
    os.makedirs(folder, exist_ok=True)
    return JSONResponse(
        status_code=200,
        content={"message": "Successfully created face", "success": True},
    )


@router.post("/faces/{name}/rename")
def rename_face(request: Request, name: str, body: dict = None):
    """Rename a face directory."""
    if not request.app.frigate_config.face_recognition.enabled:
        return JSONResponse(
            status_code=400,
            content={"message": "Face recognition is not enabled.", "success": False},
        )

    json: dict[str, any] = body or {}
    new_name = json.get("new_name")

    if not new_name:
        return JSONResponse(
            status_code=400,
            content={"message": "New name is required", "success": False},
        )

    old_folder = os.path.join(FACE_DIR, name)
    new_folder = os.path.join(FACE_DIR, new_name)

    if not os.path.exists(old_folder):
        return JSONResponse(
            status_code=404,
            content={"message": f"Face '{name}' not found", "success": False},
        )

    if os.path.exists(new_folder):
        return JSONResponse(
            status_code=400,
            content={"message": f"Face '{new_name}' already exists", "success": False},
        )

    try:
        try:
            os.rename(old_folder, new_folder)
        except OSError:
            shutil.copytree(old_folder, new_folder)
            shutil.rmtree(old_folder)

        context: EmbeddingsContext = request.app.embeddings
        context.clear_face_classifier()

        return JSONResponse(
            status_code=200,
            content={"message": "Successfully renamed face", "success": True},
        )
    except Exception as e:
        logger.error(f"Failed to rename face: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": f"Failed to rename face: {str(e)}", "success": False},
        )
