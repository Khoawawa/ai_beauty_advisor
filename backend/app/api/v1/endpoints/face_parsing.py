from fastapi import APIRouter, Response, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from core.analysis import analyze_personal_color
import cv2
import numpy as np
import traceback

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
        print(cv2_image.shape)
        stat = analyze_personal_color(cv2_image, engine)
        
        return JSONResponse(content={
            "personal_color": stat["personal_color"],
            "skintone":       stat["skintone"],
            "undertone":      stat["undertone"],
            "contrast":       stat["contrast"],
            "depth":          stat["depth"],
            "chroma":         stat["chroma"],
            # "raw_data":       stat["raw_data"],
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=traceback.format_exc())