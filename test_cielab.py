import cv2
import numpy as np
import os
import math
from advisor.face_parsing.onnx_inference import FaceParsingONNX

# ==========================================
# 1. BỘ QUY TẮC PERSONAL COLOR (BUSINESS RULES)
# ==========================================

def classify_skin_tone(l_star: float) -> str:
    if l_star > 75: return "Very light (Da rất sáng)"
    elif l_star >= 68: return "Light (Da sáng)"
    elif l_star >= 60: return "Medium light (Da sáng vừa)"
    elif l_star >= 50: return "Medium / Tan (Da trung bình/ngăm nhẹ)"
    else: return "Deep (Da ngăm/tối)"

def classify_undertone(a_star: float, b_star: float) -> str:
    if a_star == 0: return "Neutral"
    ratio = b_star / a_star
    if ratio >= 1.6: return "Warm"
    elif ratio >= 1.3: return "Neutral-warm"
    elif ratio >= 1.0: return "Neutral"
    elif ratio >= 0.8: return "Neutral-cool"
    else: return "Cool / Rosy"

def classify_contrast(skin_l: float, hair_l: float) -> str:
    delta_l = abs(skin_l - hair_l)
    if delta_l < 20: return "Low contrast (Tổng thể mềm, ít chênh lệch)"
    elif delta_l <= 35: return "Medium contrast (Tương phản vừa)"
    else: return "High contrast (Chênh rõ)"

def classify_depth(skin_l: float, hair_l: float, eye_l: float) -> str:
    score = 0.5 * (100 - hair_l) + 0.3 * (100 - eye_l) + 0.2 * (100 - skin_l)
    if score < 35: return "Light"
    elif score <= 55: return "Medium"
    else: return "Deep"

def classify_chroma(skin_c: float, lip_c: float) -> str:
    avg_c = (skin_c + lip_c) / 2
    if avg_c < 20: return "Muted"
    elif avg_c <= 32: return "Medium"
    else: return "Clear / Bright"


# ==========================================
# 2. HÀM CÂN BẰNG TRẮNG BẰNG MẮT (VON KRIES)
# ==========================================

def apply_sclera_white_balance(raw_image: np.ndarray, parsing_mask: np.ndarray, eye_classes=[4, 5]) -> np.ndarray:
    eye_mask = np.isin(parsing_mask, eye_classes).astype(np.uint8) * 255
    if cv2.countNonZero(eye_mask) == 0:
        return raw_image

    b, g, r = cv2.split(raw_image.astype(np.float32))
    eye_b, eye_g, eye_r = b[eye_mask == 255], g[eye_mask == 255], r[eye_mask == 255]
    luminance = 0.299 * eye_r + 0.587 * eye_g + 0.114 * eye_b
    
    lower_thresh = np.percentile(luminance, 40)
    upper_thresh = np.percentile(luminance, 98)
    clean_mask = (luminance >= lower_thresh) & (luminance <= upper_thresh)
    
    if not np.any(clean_mask): return raw_image
        
    b_avg, g_avg, r_avg = np.mean(eye_b[clean_mask]), np.mean(eye_g[clean_mask]), np.mean(eye_r[clean_mask])
    target_lum = (b_avg + g_avg + r_avg) / 3
    if target_lum < 30: return raw_image
        
    b_corr = np.clip(b * (target_lum / b_avg if b_avg > 0 else 1), 0, 255)
    g_corr = np.clip(g * (target_lum / g_avg if g_avg > 0 else 1), 0, 255)
    r_corr = np.clip(r * (target_lum / r_avg if r_avg > 0 else 1), 0, 255)
    
    return cv2.merge((b_corr, g_corr, r_corr)).astype(np.uint8)


# ==========================================
# 3. THỰC THI PIPELINE HOÀN CHỈNH
# ==========================================

model_path = os.path.join("advisor", "face_parsing", "weights", "resnet18.onnx")
engine = FaceParsingONNX(model_path)

image_path = "test_face.jpg"
print(f"Loading {image_path}...")
raw_image = cv2.imread(image_path)

if raw_image is None:
    print("❌ Error: Could not find the image.")
else:
    print("🤖 Running BiSeNet AI...")
    mask = engine.predict(raw_image) 
    
    print("💡 Fixing Lighting (Sclera White Balance)...")
    # QUAN TRỌNG: Sửa ánh sáng TRƯỚC khi cắt ảnh
    corrected_image = apply_sclera_white_balance(raw_image, mask, eye_classes=[4, 5])

    print("✂️ Slicing facial features...")
    background_mask = (mask == 0).astype(np.uint8) * 255
    skin_mask = (mask == 1).astype(np.uint8) * 255
    nose_mask = (mask == 2).astype(np.uint8) * 255
    lip_mask = np.isin(mask, [12, 13]).astype(np.uint8) * 255 
    hair_mask = (mask == 17).astype(np.uint8) * 255 
    facial_mask = (~np.isin(mask,[0,16])).astype(np.uint8) * 255
    eye_mask = np.isin(mask, [4, 5]).astype(np.uint8) * 255

    # Dùng ảnh ĐÃ ĐƯỢC CÂN BẰNG TRẮNG để lưu
    color_skin = cv2.bitwise_and(corrected_image, corrected_image, mask=skin_mask)
    color_lips = cv2.bitwise_and(corrected_image, corrected_image, mask=lip_mask)
    color_hair = cv2.bitwise_and(corrected_image, corrected_image, mask=hair_mask)
    color_bg = cv2.bitwise_and(raw_image, raw_image, mask=background_mask) # BG giữ nguyên raw
    color_face = cv2.bitwise_and(corrected_image, corrected_image, mask=facial_mask)
    color_eye = cv2.bitwise_and(corrected_image, corrected_image, mask=eye_mask)

    os.makedirs("extraction_results", exist_ok=True)
    cv2.imwrite("extraction_results/1_isolated_skin.jpg", color_skin)
    cv2.imwrite("extraction_results/4_isolated_bg.jpg", color_bg)
    cv2.imwrite("extraction_results/2_isolated_lips.jpg", color_lips)
    cv2.imwrite("extraction_results/3_isolated_hair.jpg", color_hair)
    cv2.imwrite("extraction_results/5_isolated_facial.jpg", color_face)
    cv2.imwrite("extraction_results/6_isolated_eyes.jpg", color_eye)
    
    # ==========================================
    # 4. CHẠY CIELAB TRÊN DỮ LIỆU ĐÃ LÀM SẠCH
    # ==========================================
    print("\n📊 Đang phân tích màu sắc CIELAB...")
    
    # Chuyển đổi toàn bộ ảnh đã sửa sáng sang CIELAB
    lab_image = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2Lab)
    l_chan, a_chan, b_chan = cv2.split(lab_image.astype(np.float32))
    
    real_l = (l_chan * 100.0) / 255.0
    real_a = a_chan - 128.0
    real_b = b_chan - 128.0

    # Hàm helper đọc mask (True/False)
    def get_cielab_avg(target_mask):
        bool_mask = target_mask == 255
        if not np.any(bool_mask): return 0.0, 0.0, 0.0, 0.0
        al = np.mean(real_l[bool_mask])
        aa = np.mean(real_a[bool_mask])
        ab = np.mean(real_b[bool_mask])
        ac = math.sqrt(aa**2 + ab**2)
        return al, aa, ab, ac

    # Lấy dữ liệu 4 vùng quan trọng
    sl, sa, sb, sc = get_cielab_avg(skin_mask)
    ll, la, lb, lc = get_cielab_avg(lip_mask)
    hl, ha, hb, hc = get_cielab_avg(hair_mask)
    el, ea, eb, ec = get_cielab_avg(eye_mask)

    print("\n🔬 --- DỮ LIỆU CIELAB TRÍCH XUẤT ---")
    print(f"Da   -> L*: {sl:.1f}, a*: {sa:.1f}, b*: {sb:.1f}, C*: {sc:.1f}")
    print(f"Môi  -> L*: {ll:.1f}, a*: {la:.1f}, b*: {lb:.1f}, C*: {lc:.1f}")
    print(f"Tóc  -> L*: {hl:.1f}, a*: {ha:.1f}, b*: {hb:.1f}, C*: {hc:.1f}")
    print(f"Mắt  -> L*: {el:.1f}, a*: {ea:.1f}, b*: {eb:.1f}, C*: {ec:.1f}")

    print("\n👑 --- KẾT QUẢ PERSONAL COLOR ANALYSIS ---")
    print(f"1. Skin Tone: {classify_skin_tone(sl)}")
    print(f"2. Undertone: {classify_undertone(sa, sb)} (Tỷ lệ b*/a*: {sb/sa if sa != 0 else 0:.2f})")
    print(f"3. Contrast:  {classify_contrast(sl, hl)} (ΔL*: {abs(sl - hl):.1f})")
    print(f"4. Depth:     {classify_depth(sl, hl, el)} (Score: {0.5*(100-hl) + 0.3*(100-el) + 0.2*(100-sl):.1f})")
    print(f"5. Chroma:    {classify_chroma(sc, lc)} (C* trung bình: {(sc + lc)/2:.1f})")

    print("\n✅ Hoàn tất toàn bộ pipeline!")