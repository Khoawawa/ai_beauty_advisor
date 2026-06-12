import cv2
import numpy as np

def apply_isolated_sclera_balance(isolated_eyes: np.ndarray, target_face: np.ndarray) -> np.ndarray:
    """
    Measures the lighting tint strictly from an isolated eye image,
    and applies the correction to the target face image.
    """
    # 1. Split the isolated eyes into channels
    eb, eg, er = cv2.split(isolated_eyes.astype(np.float32))
    
    # Calculate luminance: Y = 0.299R + 0.587G + 0.114B
    luminance = 0.299 * er + 0.587 * eg + 0.114 * eb
    
    # 2. IGNORE the pure black void
    valid_pixels_mask = luminance > 0
    if not np.any(valid_pixels_mask):
        print("⚠️ Warning: Eye image is completely empty.")
        return target_face
        
    valid_lum = luminance[valid_pixels_mask]
    
    # 3. FILTER OUT PUPIL/IRIS AND GLARE
    # Drop the darkest 40% (pupil) and brightest 2% (glare)
    lower_thresh = np.percentile(valid_lum, 40)
    upper_thresh = np.percentile(valid_lum, 98)
    
    # Create the final clean sclera mask
    clean_sclera_mask = valid_pixels_mask & (luminance >= lower_thresh) & (luminance <= upper_thresh)
    
    if not np.any(clean_sclera_mask):
        return target_face
        
    # 4. Calculate the average tint of the clean eye whites
    b_avg = np.mean(eb[clean_sclera_mask])
    g_avg = np.mean(eg[clean_sclera_mask])
    r_avg = np.mean(er[clean_sclera_mask])
    
    print(f"👁️ Sclera White Point Found -> B: {b_avg:.1f}, G: {g_avg:.1f}, R: {r_avg:.1f}")
    
    # Calculate natural brightness
    target_lum = (b_avg + g_avg + r_avg) / 3
    if target_lum < 30 or b_avg == 0 or g_avg == 0 or r_avg == 0:
        return target_face
        
    # 5. Apply the correction to the TARGET FACE
    fb, fg, fr = cv2.split(target_face.astype(np.float32))
    
    fb_corrected = np.clip(fb * (target_lum / b_avg), 0, 255)
    fg_corrected = np.clip(fg * (target_lum / g_avg), 0, 255)
    fr_corrected = np.clip(fr * (target_lum / r_avg), 0, 255)
    
    return cv2.merge((fb_corrected, fg_corrected, fr_corrected)).astype(np.uint8)


# --- DIRECT LOCAL EXECUTION ---
eye_path = "extraction_results/6_isolated_eyes.jpg"
face_path = "extraction_results/5_isolated_facial.jpg" 

print(f"Loading files...")
isolated_eyes = cv2.imread(eye_path)
isolated_face = cv2.imread(face_path)

if isolated_eyes is not None and isolated_face is not None:
    print("Executing Sclera-Anchored White Balance...")
    
    # Calculate using eyes, apply to face
    corrected_face = apply_isolated_sclera_balance(isolated_eyes, isolated_face)
    
    output_path = "sclera_balance_result.jpg"
    cv2.imwrite(output_path, corrected_face)
    print(f"✅ Success! Generated '{output_path}'. Your face is now perfectly color-balanced using your eyes as the anchor.")
else:
    print(f"❌ Error: Could not find '{eye_path}' or '{face_path}'.")