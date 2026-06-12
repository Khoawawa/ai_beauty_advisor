import cv2
import numpy as np

def apply_isolated_white_patch(isolated_bg: np.ndarray, target_image: np.ndarray = None) -> np.ndarray:
    """
    Calculates the white balance anchor from the isolated background,
    and applies it to a target image.
    """
    # If no target is provided, just apply the fix to the background image itself for testing
    if target_image is None:
        target_image = isolated_bg.copy()

    # 1. Split the isolated background into channels
    b, g, r = cv2.split(isolated_bg.astype(np.float32))
    
    # 2. Calculate luminance
    luminance = b + g + r
    
    # 3. IGNORE all the pure black pixels (where the face used to be)
    valid_pixels_mask = luminance > 0
    
    if not np.any(valid_pixels_mask):
        print("⚠️ Warning: Background image is completely black.")
        return target_image
        
    valid_luminance = luminance[valid_pixels_mask]
    
    # 4. Find the brightest 1% of the VALID background pixels
    brightest_threshold = np.percentile(valid_luminance, 99)
    
    # Create a mask of pixels that are BOTH bright and not the black void
    top_pixels_mask = (luminance >= brightest_threshold) & valid_pixels_mask
    
    b_max = np.mean(b[top_pixels_mask])
    g_max = np.mean(g[top_pixels_mask])
    r_max = np.mean(r[top_pixels_mask])
    
    print(f"🔍 Anchor Found! Brightest background color: B={b_max:.1f}, G={g_max:.1f}, R={r_max:.1f}")
    
    # Failsafe: if the background is super dark, don't break the math
    target_lum = max(b_max, g_max, r_max)
    if target_lum < 50: 
        return target_image
        
    # 5. Apply the correction to the TARGET IMAGE (Preserving original exposure)
    tb, tg, tr = cv2.split(target_image.astype(np.float32))
    
    b_corrected = np.clip(tb * (target_lum / b_max), 0, 255)
    g_corrected = np.clip(tg * (target_lum / g_max), 0, 255)
    r_corrected = np.clip(tr * (target_lum / r_max), 0, 255)
    
    return cv2.merge((b_corrected, g_corrected, r_corrected)).astype(np.uint8)


# --- TEST EXECUTION ---
bg_path = "extraction_results/4_isolated_bg.jpg"
facial_path = "extraction_results/5_isolated_facial.jpg"
print(f"Loading {bg_path}...")
isolated_bg = cv2.imread(bg_path)

print(f"Loading {facial_path}...")
isolated_face = cv2.imread(facial_path)

if isolated_bg is not None:
    print("Applying Background White Patch...")
    
    # We pass the isolated_bg as BOTH the anchor calculator and the target to test it
    corrected_bg = apply_isolated_white_patch(isolated_bg, isolated_face)
    
    output_path = "white_patch_result.jpg"
    cv2.imwrite(output_path, corrected_bg)
    print(f"✅ Success! Open '{output_path}'. The wall/AC unit should be perfectly neutral.")
else:
    print("❌ Error: Could not find the image.")

    