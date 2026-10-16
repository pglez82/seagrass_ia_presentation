import numpy as np
from PIL import Image, ImageFilter

def export_pixel_vs_dl_panels():
    panel_w = 400
    panel_h = 320
    np.random.seed(42)
    X, Y = np.meshgrid(np.linspace(0, 1, panel_w), np.linspace(0, 1, panel_h))
    
    # Smooth organic meadow shape
    dist1 = np.sqrt((X - 0.35)**2 + (Y - 0.5)**2)
    dist2 = np.sqrt((X - 0.75)**2 + (Y - 0.4)**2)
    meadow_mask = (dist1 < 0.28) | (dist2 < 0.22) | ((X > 0.1) & (X < 0.85) & (Y > 0.45) & (Y < 0.75))
    rock_mask = ((X - 0.85)**2 + (Y - 0.8)**2 < 0.015)
    meadow_mask = meadow_mask & (~rock_mask)
    
    # Panel A: Synthetic Raw RGB Image
    raw_img = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    sand_noise = np.random.normal(0, 8, (panel_h, panel_w))
    raw_img[:, :, 0] = np.clip(195 + sand_noise, 0, 255)
    raw_img[:, :, 1] = np.clip(180 + sand_noise, 0, 255)
    raw_img[:, :, 2] = np.clip(140 + sand_noise, 0, 255)

    meadow_noise = np.random.normal(0, 12, (panel_h, panel_w))
    raw_img[meadow_mask, 0] = np.clip(30 + meadow_noise[meadow_mask], 0, 255)
    raw_img[meadow_mask, 1] = np.clip(95 + meadow_noise[meadow_mask], 0, 255)
    raw_img[meadow_mask, 2] = np.clip(55 + meadow_noise[meadow_mask], 0, 255)

    raw_img[rock_mask, 0] = 75
    raw_img[rock_mask, 1] = 65
    raw_img[rock_mask, 2] = 60

    caustics = (np.sin(X * 40 + Y * 20) * 10 + np.cos(X * 30 - Y * 30) * 10).astype(np.int16)
    raw_img[:, :, 0] = np.clip(raw_img[:, :, 0] * 0.75 + caustics, 0, 255)
    raw_img[:, :, 1] = np.clip(raw_img[:, :, 1] * 0.95 + 15 + caustics, 0, 255)
    raw_img[:, :, 2] = np.clip(raw_img[:, :, 2] * 1.1 + 35 + caustics, 0, 255)

    p1 = Image.fromarray(raw_img)
    p1.save('/home/pablo/Proyectos/cartograss26/figures/panel_input.png')

    # Panel B: Pixel Classifier (RF/SVM)
    pixel_cls = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    pixel_cls[:, :] = [235, 225, 190]
    pixel_cls[meadow_mask] = [34, 112, 70]
    noise_flip = np.random.rand(panel_h, panel_w) < 0.14
    pixel_cls[noise_flip & meadow_mask] = [235, 225, 190]
    pixel_cls[noise_flip & (~meadow_mask)] = [34, 112, 70]
    pixel_cls[rock_mask] = [34, 112, 70]

    p2 = Image.fromarray(pixel_cls)
    p2.save('/home/pablo/Proyectos/cartograss26/figures/panel_pixel_rf.png')

    # Panel C: Deep Learning (U-Net)
    dl_cls = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    dl_cls[:, :] = [235, 225, 190]
    dl_cls[meadow_mask] = [34, 112, 70]
    p3 = Image.fromarray(dl_cls).filter(ImageFilter.SMOOTH)
    p3.save('/home/pablo/Proyectos/cartograss26/figures/panel_unet.png')

def export_mae_panels():
    panel_w = 300
    panel_h = 240
    np.random.seed(99)
    X, Y = np.meshgrid(np.linspace(0, 1, panel_w), np.linspace(0, 1, panel_h))
    base = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    base[:, :, 0] = np.clip(15 + X*30 + np.random.normal(0, 4, (panel_h, panel_w)), 0, 255)
    base[:, :, 1] = np.clip(110 - Y*40 + np.random.normal(0, 5, (panel_h, panel_w)), 0, 255)
    base[:, :, 2] = np.clip(150 + Y*60 + np.random.normal(0, 5, (panel_h, panel_w)), 0, 255)

    patch1 = ((X - 0.4)**2 + (Y - 0.5)**2 < 0.06)
    patch2 = ((X - 0.7)**2 + (Y - 0.3)**2 < 0.04)
    patch_all = patch1 | patch2
    base[patch_all, 0] = np.clip(base[patch_all, 0] * 0.4 + 10, 0, 255)
    base[patch_all, 1] = np.clip(base[patch_all, 1] * 0.8 + 20, 0, 255)
    base[patch_all, 2] = np.clip(base[patch_all, 2] * 0.4 + 20, 0, 255)

    p1 = Image.fromarray(base)
    p1.save('/home/pablo/Proyectos/cartograss26/figures/mae_input.png')

    grid_size = 20
    p2_arr = base.copy()
    np.random.seed(123)
    for gy in range(0, panel_h, grid_size):
        for gx in range(0, panel_w, grid_size):
            if np.random.rand() < 0.75:
                p2_arr[gy:gy+grid_size, gx:gx+grid_size] = [35, 40, 48]

    p2 = Image.fromarray(p2_arr)
    p2.save('/home/pablo/Proyectos/cartograss26/figures/mae_masked.png')

    p3 = Image.fromarray(base).filter(ImageFilter.GaussianBlur(radius=1.2))
    p3.save('/home/pablo/Proyectos/cartograss26/figures/mae_recon.png')

if __name__ == '__main__':
    export_pixel_vs_dl_panels()
    export_mae_panels()
    print("Exported clean individual panels for Beamer integration.")

