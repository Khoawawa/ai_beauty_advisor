from fastapi import APIRouter, Response, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
import cv2
import numpy as np

router = APIRouter()

@router.post("/parse-face")
async def parse_face(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    try:
        engine = request.app.state.face_parsing_engine

        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        cv2_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if cv2_image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        mask = engine.predict(cv2_image)
        # separate mask
        skin_mask = (mask == 1).astype(np.uint8) * 255
        facial_mask = (~np.isin(mask,[0,12])).astype(np.uint8) * 255
        # nose_mask = (mask == 2).astype(np.uint8) * 255
        # lip_mask = np.isin(mask, [3, 4]).astype(np.uint8) * 255
        # hair_mask = (mask == 5).astype(np.uint8) * 255
        # eye_mask = np.isin(mask, [6, 7]).astype(np.uint8) * 255

        success, encoded_mask = cv2.imencode('.png', mask)
        if not success:
            raise ValueError("Failed to encode mask image")
        
        return Response(content=encoded_mask.tobytes(), media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))