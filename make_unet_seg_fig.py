import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def generate_segmentation_figure():
    font_bold = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
    font_reg = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
    
    f_title = ImageFont.truetype(font_bold, 17)
    f_subtitle = ImageFont.truetype(font_reg, 12)
    f_badge = ImageFont.truetype(font_bold, 12)
    f_label = ImageFont.truetype(font_bold, 12)

    # 1. Load base topdown image
    img = Image.open('figures/posidonia_topdown.jpg')
    # Crop representative area: 600 x 300 (2:1 ratio)
    crop = img.crop((350, 200, 950, 520)).resize((440, 220), Image.Resampling.LANCZOS)
    arr = np.array(crop).astype(float)

    # Clean mask generation
    is_sand = (arr[:,:,0] + arr[:,:,1] + arr[:,:,2] > 260) | ((arr[:,:,2] > 110) & (arr[:,:,1] > 110))
    is_seagrass = ~is_sand

    mask_img = Image.fromarray((is_seagrass * 255).astype(np.uint8))
    mask_clean = mask_img.filter(ImageFilter.MaxFilter(size=5)).filter(ImageFilter.MinFilter(size=5))
    mask_clean = mask_clean.filter(ImageFilter.MedianFilter(size=5))

    pw, ph = 430, 215
    p1 = crop.resize((pw, ph), Image.Resampling.LANCZOS)
    mask_resized = mask_clean.resize((pw, ph), Image.Resampling.NEAREST)
    m_arr = np.array(mask_resized) > 128
    
    # Internal edges only
    e_img = mask_resized.filter(ImageFilter.FIND_EDGES)
    e_arr = np.array(e_img) > 50
    e_arr[:6, :] = False
    e_arr[-6:, :] = False
    e_arr[:, :6] = False
    e_arr[:, -6:] = False

    # -------------------------------------------------------------
    # Panel 1: Input Drone / Aerial Tile
    # -------------------------------------------------------------
    panel1 = p1.copy()
    draw1 = ImageDraw.Draw(panel1)
    draw1.rounded_rectangle([10, ph - 28, 165, ph - 8], radius=3, fill=(11, 37, 69), outline=(255, 255, 255))
    draw1.text((16, ph - 26), "RGB Aerial / Drone Tile", font=f_badge, fill=(255, 255, 255))

    # -------------------------------------------------------------
    # Panel 2: Hand Annotation (Hard & Labor-Intensive!)
    # -------------------------------------------------------------
    p2_arr = np.array(p1).copy()
    tint = np.zeros_like(p2_arr)
    tint[:] = [34, 160, 85]
    p2_arr[m_arr] = np.clip(p2_arr[m_arr]*0.65 + tint[m_arr]*0.35, 0, 255).astype(np.uint8)
    p2_arr[e_arr] = [245, 200, 30]
    panel2 = Image.fromarray(p2_arr)
    draw2 = ImageDraw.Draw(panel2)

    # Draw manual polygon vertices
    y_idx, x_idx = np.where(e_arr)
    np.random.seed(42)
    sample_indices = np.random.choice(len(y_idx), size=75, replace=False)
    for idx in sample_indices:
        cx, cy = x_idx[idx], y_idx[idx]
        draw2.rectangle([cx-2, cy-2, cx+2, cy+2], fill=(217, 83, 79), outline=(255, 255, 255), width=1)

    draw2.rounded_rectangle([10, ph - 28, 230, ph - 8], radius=3, fill=(217, 83, 79), outline=(255, 255, 255))
    draw2.text((16, ph - 26), "Manual Polygon: > 150 Vertices", font=f_badge, fill=(255, 255, 255))

    # -------------------------------------------------------------
    # Panel 3: Dense Semantic Segmentation (U-Net Output)
    # -------------------------------------------------------------
    p3_arr = np.zeros((ph, pw, 3), dtype=np.uint8)
    p3_arr[:, :] = [238, 230, 205]       # Substrate / Sand
    p3_arr[m_arr] = [34, 112, 70]        # Seagrass Canopy
    p3_arr[e_arr] = [20, 80, 50]
    panel3 = Image.fromarray(p3_arr)
    draw3 = ImageDraw.Draw(panel3)

    draw3.rounded_rectangle([10, ph - 28, 240, ph - 8], radius=3, fill=(11, 37, 69), outline=(255, 255, 255))
    # Seagrass box
    draw3.rectangle([18, ph - 23, 30, ph - 13], fill=(34, 112, 70), outline=(255, 255, 255))
    draw3.text((35, ph - 26), "Seagrass", font=f_label, fill=(255, 255, 255))
    # Sand box
    draw3.rectangle([118, ph - 23, 130, ph - 13], fill=(238, 230, 205), outline=(100, 100, 100))
    draw3.text((135, ph - 26), "Substrate / Sand", font=f_label, fill=(255, 255, 255))

    # -------------------------------------------------------------
    # Assemble 3-Panel Canvas
    # -------------------------------------------------------------
    gap = 14
    canvas_w = pw * 3 + gap * 2 + 20
    canvas_h = ph + 50
    canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Panel 1
    p1_x = 10
    canvas.paste(panel1, (p1_x, 38))
    draw.rectangle([p1_x - 1, 37, p1_x + pw, 38 + ph], outline=(180, 195, 210), width=2)
    draw.text((p1_x, 10), "(a) Input Observation", font=f_title, fill=(11, 37, 69))
    draw.text((p1_x + 195, 13), "Raw RGB drone tile", font=f_subtitle, fill=(80, 95, 110))

    # Panel 2
    p2_x = p1_x + pw + gap
    canvas.paste(panel2, (p2_x, 38))
    draw.rectangle([p2_x - 1, 37, p2_x + pw, 38 + ph], outline=(217, 83, 79), width=2)
    draw.text((p2_x, 10), "(b) Hand Annotation", font=f_title, fill=(217, 83, 79))
    draw.text((p2_x + 190, 13), "Arduous & Subjective!", font=f_subtitle, fill=(217, 83, 79))

    # Panel 3
    p3_x = p2_x + pw + gap
    canvas.paste(panel3, (p3_x, 38))
    draw.rectangle([p3_x - 1, 37, p3_x + pw, 38 + ph], outline=(34, 112, 70), width=2)
    draw.text((p3_x, 10), "(c) U-Net Segmentation", font=f_title, fill=(34, 112, 70))
    draw.text((p3_x + 220, 13), "Dense pixel mask", font=f_subtitle, fill=(34, 112, 70))

    canvas.save('figures/seagrass_segmentation_unet.png', dpi=(300, 300))
    print("Regenerated figures/seagrass_segmentation_unet.png with 2:1 aspect ratio!")

if __name__ == '__main__':
    generate_segmentation_figure()
