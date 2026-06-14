import cv2
import numpy as np
import math
from enum import Enum

class Skintone(Enum):
    VERY_LIGHT = "Very light"
    LIGHT = "Light"
    MEDIUM_LIGHT = "Medium light"
    MEDIUM = "Medium / Tan"
    DEEP = "Deep"

class Undertone(Enum):
    WARM = "Warm"
    NEUTRAL_WARM = "Neutral-warm"
    NEUTRAL = "Neutral"
    NEUTRAL_COOL = "Neutral-cool"
    COOL = "Cool / Rosy"

class Contrast(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class Depth(Enum):
    LIGHT = "Light"
    MEDIUM = "Medium"
    DEEP = "Deep"

class Chroma(Enum):
    MUTED = "Muted"
    MEDIUM = "Medium"
    BRIGHT = "Clear / Bright"

SEASON_MAP = {
    # (Undertone-group, Depth, Chroma, Contrast)
    # Springs - Warm + Clear
    (Undertone.WARM,         Depth.LIGHT,  Chroma.BRIGHT,  Contrast.LOW):    "Light Spring",
    (Undertone.WARM,         Depth.MEDIUM, Chroma.BRIGHT,  Contrast.MEDIUM): "True Spring",
    (Undertone.NEUTRAL_WARM, Depth.MEDIUM, Chroma.BRIGHT,  Contrast.HIGH):   "Bright Spring",

    # Summers - Cool + Muted
    (Undertone.COOL,         Depth.LIGHT,  Chroma.MUTED,   Contrast.LOW):    "Light Summer",
    (Undertone.COOL,         Depth.MEDIUM, Chroma.MUTED,   Contrast.LOW):    "True Summer",
    (Undertone.NEUTRAL_COOL, Depth.MEDIUM, Chroma.MUTED,   Contrast.LOW):    "Soft Summer",

    # Autumns - Warm + Muted
    (Undertone.NEUTRAL_WARM, Depth.MEDIUM, Chroma.MUTED,   Contrast.LOW):    "Soft Autumn",
    (Undertone.WARM,         Depth.MEDIUM, Chroma.MUTED,   Contrast.MEDIUM): "True Autumn",
    (Undertone.WARM,         Depth.DEEP,   Chroma.MEDIUM,  Contrast.HIGH):   "Deep Autumn",

    # Winters - Cool + Clear
    (Undertone.COOL,         Depth.DEEP,   Chroma.BRIGHT,  Contrast.HIGH):   "True Winter",
    (Undertone.NEUTRAL,      Depth.DEEP,   Chroma.BRIGHT,  Contrast.HIGH):   "Deep Winter",
    (Undertone.NEUTRAL_COOL, Depth.MEDIUM, Chroma.BRIGHT,  Contrast.HIGH):   "Bright Winter",
}

def classify_skintone(l_star: float) -> Skintone:
    if l_star > 75: return Skintone.VERY_LIGHT
    if l_star >= 68: return Skintone.LIGHT
    if l_star >= 60: return Skintone.MEDIUM_LIGHT
    if l_star >= 50: return Skintone.MEDIUM
    return Skintone.DEEP

def classify_undertone(a_star: float, b_star: float) -> Undertone:
    if a_star == 0: return Undertone.NEUTRAL
    ratio = b_star / a_star
    if ratio >= 1.6: return Undertone.WARM
    elif ratio >= 1.3: return Undertone.NEUTRAL_WARM
    elif ratio >= 1.0: return Undertone.NEUTRAL
    elif ratio >= 0.8: return Undertone.NEUTRAL_COOL
    return Undertone.COOL

def classify_contrast(skin_l: float, hair_l: float) -> Contrast:
    delta_l = abs(skin_l - hair_l)
    if delta_l < 20: return Contrast.LOW
    elif delta_l <= 35: return Contrast.MEDIUM
    return Contrast.HIGH

def classify_depth(skin_l: float, hair_l: float, eye_l: float) -> Depth:
    score = 0.5 * (100 - hair_l) + 0.3 * (100 - eye_l) + 0.2 * (100 - skin_l)
    if score < 35: return Depth.LIGHT
    elif score <= 55: return Depth.MEDIUM
    return Depth.DEEP

def classify_chroma(skin_c: float, lip_c: float) -> Chroma:
    avg_c = (skin_c + lip_c) / 2
    if avg_c < 20: return Chroma.MUTED
    elif avg_c <= 32: return Chroma.MEDIUM
    return Chroma.BRIGHT

def apply_sclera_white_balance(raw_image: np.ndarray, parsing_mask: np.ndarray, eye_classes=[4, 5]) -> np.ndarray:
    # normalize image using the whiteness of the sclera (eye white)
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

def analyze_personal_color(image_bgr: np.ndarray, engine) -> dict:
    stat = get_stat(image_bgr, engine)
    print("This work")
    
    undertone_enum = next(u for u in Undertone if u.value == stat["undertone"])
    depth_enum     = next(d for d in Depth     if d.value == stat["depth"])
    chroma_enum    = next(c for c in Chroma    if c.value == stat["chroma"])
    contrast_enum  = next(c for c in Contrast  if c.value == stat["contrast"])

    season = get_personal_color(undertone_enum, depth_enum, chroma_enum, contrast_enum)
    print("this work too")
    return {
        **stat,
        "personal_color": season,
    }
def get_personal_color(undertone: Undertone, depth: Depth, chroma: Chroma, contrast: Contrast) -> str:
    # 1. Try exact match first
    key = (undertone, depth, chroma, contrast)
    if key in SEASON_MAP:
        return SEASON_MAP[key]

    # 2. Score each season by how many dimensions match
    UNDERTONE_WARM  = {Undertone.WARM, Undertone.NEUTRAL_WARM}
    UNDERTONE_COOL  = {Undertone.COOL, Undertone.NEUTRAL_COOL}
    UNDERTONE_NEUT  = {Undertone.NEUTRAL}

    SEASON_PROFILES = {
        "Light Spring":   (UNDERTONE_WARM, Depth.LIGHT,  Chroma.BRIGHT,  Contrast.LOW),
        "True Spring":    (UNDERTONE_WARM, Depth.MEDIUM, Chroma.BRIGHT,  Contrast.MEDIUM),
        "Bright Spring":  (UNDERTONE_WARM | UNDERTONE_NEUT, Depth.MEDIUM, Chroma.BRIGHT, Contrast.HIGH),
        "Light Summer":   (UNDERTONE_COOL, Depth.LIGHT,  Chroma.MUTED,   Contrast.LOW),
        "True Summer":    (UNDERTONE_COOL, Depth.MEDIUM, Chroma.MUTED,   Contrast.LOW),
        "Soft Summer":    (UNDERTONE_COOL | UNDERTONE_NEUT, Depth.MEDIUM, Chroma.MUTED, Contrast.LOW),
        "Soft Autumn":    (UNDERTONE_WARM | UNDERTONE_NEUT, Depth.MEDIUM, Chroma.MUTED, Contrast.LOW),
        "True Autumn":    (UNDERTONE_WARM, Depth.MEDIUM, Chroma.MUTED,   Contrast.MEDIUM),
        "Deep Autumn":    (UNDERTONE_WARM, Depth.DEEP,   Chroma.MEDIUM,  Contrast.HIGH),
        "True Winter":    (UNDERTONE_COOL, Depth.DEEP,   Chroma.BRIGHT,  Contrast.HIGH),
        "Deep Winter":    (UNDERTONE_COOL | UNDERTONE_NEUT, Depth.DEEP,  Chroma.BRIGHT, Contrast.HIGH),
        "Bright Winter":  (UNDERTONE_COOL | UNDERTONE_NEUT, Depth.MEDIUM, Chroma.BRIGHT, Contrast.HIGH),
    }

    best_season, best_score = "True Spring", -1
    for season, (ut_set, dep, chr_, con) in SEASON_PROFILES.items():
        score = 0
        score += 3 if undertone in ut_set else 0   # undertone weighted most
        score += 2 if depth == dep else 0
        score += 2 if chroma == chr_ else 0
        score += 1 if contrast == con else 0
        if score > best_score:
            best_score, best_season = score, season

    return best_season
def apply_gray_world(image_bgr: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(image_bgr.astype(np.float32))
    b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
    gray = (b_mean + g_mean + r_mean) / 3
    b = np.clip(b * (gray / b_mean), 0, 255)
    g = np.clip(g * (gray / g_mean), 0, 255)
    r = np.clip(r * (gray / r_mean), 0, 255)
    return cv2.merge((b, g, r)).astype(np.uint8)
def get_stat(image_bgr: np.ndarray, engine) -> dict:

    
    mask = engine.predict(image_bgr) 
    
    # 2. Lighting Correction
    corrected = apply_gray_world(image_bgr)
    
    # 3. Convert to Scientific CIELAB Space
    lab_image = cv2.cvtColor(corrected, cv2.COLOR_BGR2Lab)
    l_chan, a_chan, b_chan = cv2.split(lab_image.astype(np.float32))
    
    real_l = (l_chan * 100.0) / 255.0
    real_a = a_chan - 128.0
    real_b = b_chan - 128.0

    # Define regions based on standard CelebAMask-HQ classes
    skin_mask = (mask == 1) | (mask == 2)
    lip_mask = np.isin(mask, [12, 13])
    hair_mask = (mask == 17)
    eye_mask = np.isin(mask, [4, 5])

    # 4. Math Helper
    def get_cielab_avg(target_mask):
        if not np.any(target_mask): 
            return 0.0, 0.0, 0.0, 0.0
        al = np.mean(real_l[target_mask])
        aa = np.mean(real_a[target_mask])
        ab = np.mean(real_b[target_mask])
        ac = math.sqrt(aa**2 + ab**2)
        return al, aa, ab, ac

    # 5. Extract Feature Colors
    sl, sa, sb, sc = get_cielab_avg(skin_mask)
    ll, la, lb, lc = get_cielab_avg(lip_mask)
    hl, ha, hb, hc = get_cielab_avg(hair_mask)
    el, ea, eb, ec = get_cielab_avg(eye_mask)

    # 6. Apply Business Logic & Return Enums as readable strings
    return {
        "skintone": classify_skintone(sl).value,
        "undertone": classify_undertone(sa, sb).value,
        "contrast": classify_contrast(sl, hl).value,
        "depth": classify_depth(sl, hl, el).value,
        "chroma": classify_chroma(sc, lc).value,
        # "raw_data": {
        #     "skin_L": round(sl, 2), "skin_a": round(sa, 2), "skin_b": round(sb, 2)
        # }
    }