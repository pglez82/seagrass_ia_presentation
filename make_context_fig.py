import numpy as np
from PIL import Image, ImageDraw, ImageFont

def generate_pixel_patch_context_figure():
    # Load fonts
    font_path_bold = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
    font_path_reg = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
    
    font_title = ImageFont.truetype(font_path_bold, 21)
    font_subtitle = ImageFont.truetype(font_path_bold, 17)
    font_body = ImageFont.truetype(font_path_reg, 15)
    font_body_bold = ImageFont.truetype(font_path_bold, 15)
    font_small = ImageFont.truetype(font_path_reg, 13)
    font_kernel = ImageFont.truetype(font_path_bold, 18)
    font_eq = ImageFont.truetype(font_path_bold, 18)

    # 1. Load base underwater image
    base_img = Image.open('/home/pablo/Proyectos/cartograss26/figures/posidonia_underwater.jpg')
    # Crop a detailed region: seagrass leaves, blades, sand
    crop_box = (200, 160, 850, 700) # (650 x 540)
    cropped_scene = base_img.crop(crop_box).resize((430, 370), Image.Resampling.LANCZOS)

    # Coordinates on the cropped scene for the patch
    patch_size = 90
    px_center = (230, 190)
    patch_box = (
        px_center[0] - patch_size // 2,
        px_center[1] - patch_size // 2,
        px_center[0] + patch_size // 2,
        px_center[1] + patch_size // 2
    )

    # Extract patch
    extracted_patch = cropped_scene.crop(patch_box)
    
    # Center pixel color
    center_color = cropped_scene.getpixel(px_center)

    # Draw markings on the scene image
    scene_annotated = cropped_scene.copy()
    draw_scene = ImageDraw.Draw(scene_annotated)
    # Patch box: green border
    draw_scene.rectangle(patch_box, outline=(34, 112, 70), width=4)
    # Center pixel box: red highlight
    cx, cy = px_center
    draw_scene.rectangle([cx-4, cy-4, cx+4, cy+4], fill=(217, 83, 79), outline=(255, 255, 255), width=1)

    # Overall canvas setup
    canvas_w = 1520
    canvas_h = 490
    bg_color = (255, 255, 255)
    canvas = Image.new('RGB', (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(canvas)

    # Panel 1: (a) Underwater Observation
    p1_x, p1_y = 25, 45
    canvas.paste(scene_annotated, (p1_x, p1_y))
    draw.rectangle([p1_x-1, p1_y-1, p1_x + 430 + 1, p1_y + 370 + 1], outline=(180, 195, 210), width=2)
    draw.text((p1_x, 14), "(a) Underwater Observation", font=font_title, fill=(11, 37, 69))
    
    # Legend below scene
    draw.rectangle([p1_x + 10, p1_y + 382, p1_x + 25, p1_y + 397], fill=(217, 83, 79))
    draw.text((p1_x + 32, p1_y + 380), "Single Pixel", font=font_small, fill=(70, 80, 95))
    draw.rectangle([p1_x + 160, p1_y + 382, p1_x + 175, p1_y + 397], outline=(34, 112, 70), width=3)
    draw.text((p1_x + 182, p1_y + 380), "Spatial Context Patch", font=font_small, fill=(70, 80, 95))

    # Panel 2: (b) Isolated Pixel
    p2_x, p2_y = 505, 45
    card_w, card_h = 270, 370
    draw.rounded_rectangle([p2_x, p2_y, p2_x + card_w, p2_y + card_h], radius=8, fill=(253, 248, 248), outline=(235, 180, 180), width=2)
    draw.text((p2_x + 15, 14), "(b) 1 Pixel (Zero Context)", font=font_title, fill=(217, 83, 79))

    # Big pixel representation (180x180)
    px_disp_size = 180
    px_disp_x = p2_x + (card_w - px_disp_size) // 2
    px_disp_y = p2_y + 20
    pixel_img = Image.new('RGB', (px_disp_size, px_disp_size), center_color)
    draw_px = ImageDraw.Draw(pixel_img)
    draw_px.rectangle([0, 0, px_disp_size-1, px_disp_size-1], outline=(217, 83, 79), width=3)
    canvas.paste(pixel_img, (px_disp_x, px_disp_y))

    # Pixel description
    draw.text((p2_x + 15, px_disp_y + 195), "RGB Spectrum Only:", font=font_subtitle, fill=(217, 83, 79))
    draw.text((p2_x + 15, px_disp_y + 225), "• Severe Spectral Overlap", font=font_body_bold, fill=(30, 40, 50))
    draw.text((p2_x + 15, px_disp_y + 250), "• Macroalgae? Rock shadow?", font=font_body, fill=(70, 80, 95))
    draw.text((p2_x + 15, px_disp_y + 275), "• Posidonia? Impossible to tell!", font=font_body_bold, fill=(217, 83, 79))

    # Panel 3: (c) Spatial Patch
    p3_x, p3_y = 825, 45
    draw.rounded_rectangle([p3_x, p3_y, p3_x + card_w, p3_y + card_h], radius=8, fill=(247, 252, 249), outline=(170, 215, 190), width=2)
    draw.text((p3_x + 15, 14), "(c) Spatial Patch (Context)", font=font_title, fill=(34, 112, 70))

    # Extracted patch resized to 180x180
    patch_disp = extracted_patch.resize((px_disp_size, px_disp_size), Image.Resampling.LANCZOS)
    draw_p = ImageDraw.Draw(patch_disp)
    draw_p.rectangle([0, 0, px_disp_size-1, px_disp_size-1], outline=(34, 112, 70), width=3)
    canvas.paste(patch_disp, (p3_x + (card_w - px_disp_size) // 2, p3_y + 20))

    # Patch description
    draw.text((p3_x + 15, px_disp_y + 195), "Morphology & Context:", font=font_subtitle, fill=(34, 112, 70))
    draw.text((p3_x + 15, px_disp_y + 225), "• Blade ribbons & orientation", font=font_body_bold, fill=(30, 40, 50))
    draw.text((p3_x + 15, px_disp_y + 250), "• Contrast with seabed sand", font=font_body, fill=(70, 80, 95))
    draw.text((p3_x + 15, px_disp_y + 275), "• Unmistakable Posidonia!", font=font_body_bold, fill=(34, 112, 70))

    # Panel 4: (d) 2D Convolution Kernel
    p4_x, p4_y = 1145, 45
    card4_w = 350
    draw.rounded_rectangle([p4_x, p4_y, p4_x + card4_w, p4_y + card_h], radius=8, fill=(246, 250, 254), outline=(170, 205, 230), width=2)
    draw.text((p4_x + 15, 14), "(d) Learned 2D Convolution", font=font_title, fill=(0, 119, 145))

    # Draw 3x3 kernel matrix grid
    k_x, k_y = p4_x + 35, p4_y + 25
    c_size = 46
    
    # Kernel box background
    draw.rectangle([k_x, k_y, k_x + 3*c_size, k_y + 3*c_size], fill=(255, 255, 255), outline=(0, 119, 145), width=2)
    for i in range(1, 3):
        draw.line([(k_x + i*c_size, k_y), (k_x + i*c_size, k_y + 3*c_size)], fill=(200, 220, 235), width=2)
        draw.line([(k_x, k_y + i*c_size), (k_x + 3*c_size, k_y + i*c_size)], fill=(200, 220, 235), width=2)

    kernel_vals = [
        ["-1", "-1", "-1"],
        ["-1", " 8", "-1"],
        ["-1", "-1", "-1"]
    ]
    for r in range(3):
        for c in range(3):
            val = kernel_vals[r][c]
            tx = k_x + c*c_size + (12 if "-" in val else 14)
            ty = k_y + r*c_size + 13
            draw.text((tx, ty), val, font=font_kernel, fill=(0, 119, 145) if val != " 8" else (217, 83, 79))

    # Convolution label beside grid
    draw.text((k_x + 3*c_size + 20, k_y + 20), "Kernel W", font=font_subtitle, fill=(0, 119, 145))
    draw.text((k_x + 3*c_size + 20, k_y + 48), "Learned filter weights", font=font_body, fill=(70, 80, 95))
    draw.text((k_x + 3*c_size + 20, k_y + 72), "Sliding over patch X", font=font_body, fill=(70, 80, 95))
    draw.text((k_x + 3*c_size + 20, k_y + 96), "Edge / ribbon detector", font=font_small, fill=(34, 112, 70))

    # Mathematical operation box
    draw.rectangle([p4_x + 20, p4_y + 185, p4_x + card4_w - 20, p4_y + 242], fill=(255, 255, 255), outline=(0, 119, 145), width=1)
    draw.text((p4_x + 35, p4_y + 193), "y = ReLU( W * X + b )", font=font_eq, fill=(11, 37, 69))
    draw.text((p4_x + 35, p4_y + 219), "Feature map captures local spatial gradients", font=font_small, fill=(0, 119, 145))

    # CNN takeaway
    draw.text((p4_x + 20, p4_y + 255), "How CNNs Learn Context:", font=font_subtitle, fill=(11, 37, 69))
    draw.text((p4_x + 20, p4_y + 285), "• Kernels automatically optimize", font=font_body_bold, fill=(30, 40, 50))
    draw.text((p4_x + 20, p4_y + 308), "  to detect seagrass texture & ribbons.", font=font_body, fill=(70, 80, 95))
    draw.text((p4_x + 20, p4_y + 332), "• Eliminates salt-and-pepper noise!", font=font_body_bold, fill=(34, 112, 70))

    # Connecting arrows between panels
    arrow_y = p1_y + 110
    
    # Arrow 1: Scene to Pixel
    draw.line([(p1_x + 430 + 6, arrow_y), (p2_x - 12, arrow_y)], fill=(217, 83, 79), width=3)
    draw.polygon([(p2_x - 4, arrow_y), (p2_x - 14, arrow_y - 6), (p2_x - 14, arrow_y + 6)], fill=(217, 83, 79))

    # Arrow 2: Pixel to Patch
    draw.line([(p2_x + card_w + 6, arrow_y), (p3_x - 12, arrow_y)], fill=(34, 112, 70), width=3)
    draw.polygon([(p3_x - 4, arrow_y), (p3_x - 14, arrow_y - 6), (p3_x - 14, arrow_y + 6)], fill=(34, 112, 70))

    # Arrow 3: Patch to Conv
    draw.line([(p3_x + card_w + 6, arrow_y), (p4_x - 12, arrow_y)], fill=(0, 119, 145), width=3)
    draw.polygon([(p4_x - 4, arrow_y), (p4_x - 14, arrow_y - 6), (p4_x - 14, arrow_y + 6)], fill=(0, 119, 145))

    canvas.save('/home/pablo/Proyectos/cartograss26/figures/pixel_vs_patch_context.png', dpi=(300, 300))
    print("Successfully generated figures/pixel_vs_patch_context.png")

if __name__ == '__main__':
    generate_pixel_patch_context_figure()

