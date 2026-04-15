"""Folders API — user-uploaded PDF collections the agent can read."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from pathclaw import folders as folders_store

router = APIRouter()


class CreateFolderRequest(BaseModel):
    name: str


class AttachRequest(BaseModel):
    session_id: str


@router.get("")
async def list_folders():
    return {"folders": folders_store.list_folders()}


@router.post("")
async def create_folder(req: CreateFolderRequest):
    if not req.name.strip():
        raise HTTPException(400, "name is required")
    return folders_store.create_folder(req.name)


@router.get("/{folder_id}")
async def get_folder(folder_id: str):
    meta = folders_store.get_folder(folder_id)
    if not meta:
        raise HTTPException(404, f"Folder {folder_id} not found")
    return meta


@router.delete("/{folder_id}")
async def delete_folder(folder_id: str):
    ok = folders_store.delete_folder(folder_id)
    if not ok:
        raise HTTPException(404, f"Folder {folder_id} not found")
    return {"ok": True}


@router.post("/{folder_id}/upload")
async def upload_pdf(folder_id: str, file: UploadFile = File(...)):
    if not folders_store.get_folder(folder_id):
        raise HTTPException(404, f"Folder {folder_id} not found")
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > 50 * 1024 * 1024:  # 50 MB cap per PDF
        raise HTTPException(413, "PDF larger than 50 MB")
    try:
        info = folders_store.save_pdf(folder_id, file.filename or "upload.pdf", data)
    except Exception as e:
        raise HTTPException(500, f"Failed to save PDF: {e}")
    return info


@router.delete("/{folder_id}/files/{filename:path}")
async def delete_file(folder_id: str, filename: str):
    ok = folders_store.delete_file(folder_id, filename)
    if not ok:
        raise HTTPException(404, f"{filename} not found")
    return {"ok": True}


@router.get("/{folder_id}/files/{filename:path}/text")
async def read_text(folder_id: str, filename: str):
    try:
        text = folders_store.read_pdf_text(folder_id, filename)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to parse PDF: {e}")
    return {"text": text, "char_count": len(text)}


@router.post("/{folder_id}/attach")
async def attach(folder_id: str, req: AttachRequest):
    if not req.session_id.strip():
        raise HTTPException(400, "session_id is required")
    ok = folders_store.attach_to_session(folder_id, req.session_id)
    if not ok:
        raise HTTPException(404, f"Folder {folder_id} not found")
    return {"ok": True}


@router.post("/{folder_id}/detach")
async def detach(folder_id: str, req: AttachRequest):
    ok = folders_store.detach_from_session(folder_id, req.session_id)
    if not ok:
        raise HTTPException(404, f"Folder {folder_id} not found")
    return {"ok": True}


@router.get("/session/{session_id}")
async def folders_for_session(session_id: str):
    return {"folders": folders_store.folders_for_session(session_id)}
