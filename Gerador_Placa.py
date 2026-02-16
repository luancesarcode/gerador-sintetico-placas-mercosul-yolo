import argparse
import os
import random
import shutil
import string
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ================= CONFIGURACAO PADRAO =================
DEFAULT_PLATE_TEMPLATE = "assets/placa_mercosul.png"
DEFAULT_FONT_PATH = "assets/fonts/FE-Schrift.TTF"
DEFAULT_NUM_IMAGES = 500

DEFAULT_IMG_W, DEFAULT_IMG_H = 1280, 720

DEFAULT_FONT_SIZE = 340
DEFAULT_TEXT_X = 260
DEFAULT_TEXT_Y = 220
DEFAULT_BACKGROUND_DIR = "assets/backgrounds"
DEFAULT_PLATE_TEMPLATES_DIR = "assets/plates"

# Classes YOLO
# 0 = placa
# 1 = caractere
# ================================================

DEFAULT_SCALE_MIN = 0.12
DEFAULT_SCALE_MAX = 0.4
DEFAULT_MAX_PLATE_WIDTH_RATIO = 0.25
DEFAULT_MAX_PLATE_HEIGHT_RATIO = 0.18
DEFAULT_PLACEMENT_MARGIN_RATIO = 0.05
ROTATION_MIN = -5.0
ROTATION_MAX = 5.0

RAIN_PROB = 0.7
NIGHT_PROB = 0.5
OCCLUSION_PROB = 0.0
BLUR_PROB = 0.4

BACKGROUND_JITTER_PROB = 0.9
BACKGROUND_TINT_PROB = 0.3
BACKGROUND_BLUR_MIN = 3
BACKGROUND_BLUR_MAX = 7

PERSPECTIVE_PROB = 0.85
PERSPECTIVE_MAX_WARP = 0.14

CURVATURE_PROB = 0.35
CURVATURE_STRENGTH_MIN = 2.0
CURVATURE_STRENGTH_MAX = 8.0

LENS_DISTORT_PROB = 0.35
LENS_K1_MIN = -0.0012
LENS_K1_MAX = 0.0012

PLATE_DIRT_PROB = 0.5
PLATE_SCRATCH_PROB = 0.4
PLATE_SCREW_PROB = 0.6
PLATE_SPECULAR_PROB = 0.4
PLATE_TINT_PROB = 0.3

CAM_BRIGHT_CONTRAST_PROB = 0.9
CAM_WHITE_BALANCE_PROB = 0.6
CAM_SHADOW_PROB = 0.4
CAM_NOISE_PROB = 0.6
CAM_JPEG_PROB = 0.5
CAM_DOWNSCALE_PROB = 0.4
CAM_SHARPEN_PROB = 0.35
CAM_VIGNETTE_PROB = 0.45
CAM_FLARE_PROB = 0.2


# ---------- TEXTO DA PLACA ----------
def gerar_texto_placa():
    return (
        "".join(random.choices(string.ascii_uppercase, k=3))
        + random.choice("0123456789")
        + random.choice(string.ascii_uppercase)
        + "".join(random.choices("0123456789", k=2))
    )


# ---------- CHUVA NA PLACA ----------
def adicionar_chuva_na_regiao(img):
    rain = np.zeros_like(img)
    for _ in range(1500):
        x = random.randint(0, img.shape[1] - 1)
        y = random.randint(0, img.shape[0] - 1)
        length = random.randint(10, 30)
        cv2.line(rain, (x, y), (x + 2, y + length), (200, 200, 200), 1)
    return cv2.addWeighted(img, 1, rain, 0.5, 0)


# ---------- NOITE ----------
def efeito_noturno(img):
    img = img.astype(np.float32)
    factor = random.choice([random.uniform(0.35, 0.6), random.uniform(0.2, 0.35)])
    img *= factor
    noise = np.random.normal(0, 20, img.shape)
    img += noise
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------- OCLUSAO ----------
def aplicar_oclusao(img):
    h, w, _ = img.shape
    for _ in range(random.randint(1, 2)):
        x = random.randint(0, w - 40)
        y = random.randint(0, h - 20)
        cv2.rectangle(
            img,
            (x, y),
            (x + random.randint(40, 120), y + random.randint(20, 60)),
            (40, 40, 40),
            -1,
        )
    return img


# ---------- ROTACAO ----------
def rotacionar_imagem(img, angle):
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h), borderValue=(120, 120, 120))
    return rotated, matrix


# ---------- DESFOQUE DE MOVIMENTO ----------
def desfoque_movimento(img):
    k = random.choice([5, 7, 9])
    kernel = np.zeros((k, k))
    kernel[k // 2, :] = np.ones(k)
    kernel /= k
    return cv2.filter2D(img, -1, kernel)


# ---------- AJUDANTES DE FUNDO / TEMPLATE ----------
def listar_arquivos_imagem(dir_path):
    if not dir_path:
        return []
    path = Path(dir_path)
    if not path.is_dir():
        return []
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    files = []
    for ext in exts:
        files.extend(path.glob(ext))
        files.extend(path.glob(ext.upper()))
    return sorted(set(files))


def redimensionar_e_recortar(img, target_w, target_h):
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.full((target_h, target_w, 3), 120, np.uint8)
    scale = max(target_w / w, target_h / h)
    new_w = max(target_w, int(round(w * scale)))
    new_h = max(target_h, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0 = 0 if new_w == target_w else random.randint(0, new_w - target_w)
    y0 = 0 if new_h == target_h else random.randint(0, new_h - target_h)
    return resized[y0 : y0 + target_h, x0 : x0 + target_w].copy()


def aplicar_jitter_fundo(img):
    alpha = random.uniform(0.85, 1.15)
    beta = random.randint(-18, 18)
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    if random.random() < BACKGROUND_TINT_PROB:
        tint_strength = random.uniform(0.05, 0.15)
        tint_color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        tint = np.full_like(out, tint_color)
        out = cv2.addWeighted(out, 1 - tint_strength, tint, tint_strength, 0)
    return out


def aplicar_desfoque_fundo(img, k_min, k_max):
    k_min = max(1, k_min)
    k_max = max(k_min, k_max)
    if k_min % 2 == 0:
        k_min += 1
    if k_max % 2 == 0:
        k_max -= 1
    if k_max < k_min:
        k_max = k_min
    if k_max <= 1:
        return img
    k = random.randrange(k_min, k_max + 1, 2)
    if k <= 1:
        return img
    return cv2.GaussianBlur(img, (k, k), 0)


def criar_fundo(target_w, target_h, bg_paths):
    if bg_paths:
        for _ in range(5):
            bg_path = random.choice(bg_paths)
            img = cv2.imread(str(bg_path))
            if img is not None:
                background = redimensionar_e_recortar(img, target_w, target_h)
                if random.random() < BACKGROUND_JITTER_PROB:
                    background = aplicar_jitter_fundo(background)
                background = aplicar_desfoque_fundo(
                    background, BACKGROUND_BLUR_MIN, BACKGROUND_BLUR_MAX
                )
                return background
    return np.full((target_h, target_w, 3), 120, np.uint8)


# ---------- APARENCIA DA PLACA ----------
def aplicar_tinta_placa(plate_rgb):
    alpha = random.uniform(0.9, 1.1)
    beta = random.randint(-10, 10)
    out = cv2.convertScaleAbs(plate_rgb, alpha=alpha, beta=beta)
    return out


def adicionar_parafusos_placa(plate_rgb):
    h, w = plate_rgb.shape[:2]
    margin_x = max(8, int(w * 0.06))
    margin_y = max(8, int(h * 0.12))
    radius = max(4, int(min(w, h) * 0.02))
    centers = [
        (margin_x, margin_y),
        (w - margin_x, margin_y),
        (margin_x, h - margin_y),
        (w - margin_x, h - margin_y),
    ]
    for (cx, cy) in centers:
        cv2.circle(plate_rgb, (cx, cy), radius, (40, 40, 40), -1)
        cv2.circle(plate_rgb, (cx - 1, cy - 1), max(1, radius // 3), (200, 200, 200), -1)
    return plate_rgb


def adicionar_sujeira_placa(plate_rgb):
    h, w = plate_rgb.shape[:2]
    dust = np.random.rand(h, w)
    mask = dust < random.uniform(0.01, 0.03)
    dirt_color = np.array(
        [
            random.randint(60, 110),
            random.randint(60, 110),
            random.randint(60, 110),
        ],
        dtype=np.uint8,
    )
    plate_rgb[mask] = (
        plate_rgb[mask].astype(np.float32) * 0.7
        + dirt_color.astype(np.float32) * 0.3
    ).astype(np.uint8)
    return plate_rgb


def adicionar_riscos_placa(plate_rgb):
    h, w = plate_rgb.shape[:2]
    for _ in range(random.randint(6, 12)):
        x1 = random.randint(0, w - 1)
        y1 = random.randint(0, h - 1)
        length = random.randint(int(w * 0.2), int(w * 0.6))
        angle = random.uniform(-0.4, 0.4)
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        color = random.randint(160, 220)
        cv2.line(plate_rgb, (x1, y1), (x2, y2), (color, color, color), 1)
    return plate_rgb


def adicionar_reflexo_especular(roi):
    h, w = roi.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    thickness = random.randint(max(6, int(h * 0.12)), max(10, int(h * 0.2)))
    x0 = random.randint(-w // 2, w)
    y0 = random.randint(-h // 2, h)
    x1 = x0 + w + random.randint(-w // 3, w // 3)
    y1 = y0 + h + random.randint(-h // 3, h // 3)
    cv2.line(mask, (x0, y0), (x1, y1), 255, thickness)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=thickness / 2)
    alpha = random.uniform(0.15, 0.35)
    highlight = (mask.astype(np.float32) / 255.0) * alpha
    roi_float = roi.astype(np.float32)
    roi_float = np.clip(roi_float + highlight[..., None] * 255.0, 0, 255)
    return roi_float.astype(np.uint8)


# ---------- TRANSFORMACOES GEOMETRICAS ----------
def redimensionar_rgba_e_mascaras(img_rgba, masks, scale):
    if abs(scale - 1.0) < 1e-6:
        return img_rgba, masks
    resized = cv2.resize(img_rgba, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    resized_masks = [
        cv2.resize(m, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        for m in masks
    ]
    return resized, resized_masks


def rotacionar_rgba_e_mascaras(img_rgba, masks, angle):
    h, w = img_rgba.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    channels = 1 if img_rgba.ndim < 3 else img_rgba.shape[2]
    border_value = 0 if channels == 1 else (0,) * channels
    rotated = cv2.warpAffine(
        img_rgba,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderValue=border_value,
    )
    rotated_masks = [
        cv2.warpAffine(m, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        for m in masks
    ]
    return rotated, rotated_masks


def deformar_perspectiva_rgba_e_mascaras(img_rgba, masks, max_warp):
    if max_warp <= 0:
        return img_rgba, masks
    h, w = img_rgba.shape[:2]
    max_dx = w * max_warp
    max_dy = h * max_warp
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32(
        [
            [random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy)],
            [w - 1 + random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy)],
            [w - 1 + random.uniform(-max_dx, max_dx), h - 1 + random.uniform(-max_dy, max_dy)],
            [random.uniform(-max_dx, max_dx), h - 1 + random.uniform(-max_dy, max_dy)],
        ]
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    channels = 1 if img_rgba.ndim < 3 else img_rgba.shape[2]
    border_value = 0 if channels == 1 else (0,) * channels
    warped = cv2.warpPerspective(
        img_rgba,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderValue=border_value,
    )
    warped_masks = [
        cv2.warpPerspective(m, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        for m in masks
    ]
    return warped, warped_masks


def aplicar_deformacao_curvatura(img_rgba, masks, strength):
    h, w = img_rgba.shape[:2]
    if h <= 1 or w <= 1 or abs(strength) < 0.5:
        return img_rgba, masks
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(xs, ys)
    if random.random() < 0.5:
        dx = strength * np.sin(np.pi * y_grid)  # ondulacao horizontal
        dy = 0.0
    else:
        dy = strength * np.sin(np.pi * x_grid)  # ondulacao vertical
        dx = 0.0
    map_x = (x_grid * (w / 2.0) + (w / 2.0) + dx).astype(np.float32)
    map_y = (y_grid * (h / 2.0) + (h / 2.0) + dy).astype(np.float32)
    channels = 1 if img_rgba.ndim < 3 else img_rgba.shape[2]
    border_value = 0 if channels == 1 else (0,) * channels
    warped = cv2.remap(
        img_rgba,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    warped_masks = [
        cv2.remap(
            m,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        for m in masks
    ]
    return warped, warped_masks


def aplicar_distorcao_lente(img_rgba, masks, k1):
    h, w = img_rgba.shape[:2]
    if abs(k1) < 1e-7:
        return img_rgba, masks
    fx = w
    fy = h
    cx = w / 2.0
    cy = h / 2.0
    xs = (np.arange(w, dtype=np.float32) - cx) / fx
    ys = (np.arange(h, dtype=np.float32) - cy) / fy
    x_grid, y_grid = np.meshgrid(xs, ys)
    r2 = x_grid * x_grid + y_grid * y_grid
    factor = 1.0 + k1 * r2
    map_x = (x_grid * factor) * fx + cx
    map_y = (y_grid * factor) * fy + cy
    channels = 1 if img_rgba.ndim < 3 else img_rgba.shape[2]
    border_value = 0 if channels == 1 else (0,) * channels
    distorted = cv2.remap(
        img_rgba,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    distorted_masks = [
        cv2.remap(
            m,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        for m in masks
    ]
    return distorted, distorted_masks


def mesclar_alpha(canvas_bgr, overlay_rgba, x, y):
    h, w = overlay_rgba.shape[:2]
    roi = canvas_bgr[y : y + h, x : x + w].astype(np.float32)
    overlay_rgb = overlay_rgba[:, :, :3].astype(np.float32)
    alpha = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0
    blended = roi * (1.0 - alpha) + overlay_rgb * alpha
    canvas_bgr[y : y + h, x : x + w] = blended.astype(np.uint8)


# ---------- EFEITOS DE CAMERA ----------
def aplicar_brilho_contraste(img):
    alpha = random.uniform(0.85, 1.2)
    beta = random.randint(-25, 25)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def aplicar_balanco_branco(img):
    gains = np.array(
        [
            random.uniform(0.9, 1.1),
            random.uniform(0.9, 1.1),
            random.uniform(0.9, 1.1),
        ],
        dtype=np.float32,
    )
    out = img.astype(np.float32) * gains
    return np.clip(out, 0, 255).astype(np.uint8)


def aplicar_sombra(img):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x1 = random.randint(-w // 3, w)
    y1 = random.randint(-h // 3, h)
    x2 = random.randint(0, w)
    y2 = random.randint(0, h)
    x3 = random.randint(0, w + w // 3)
    y3 = random.randint(0, h + h // 3)
    polygon = np.array([[x1, y1], [x2, y2], [x3, y3]], dtype=np.int32)
    cv2.fillConvexPoly(mask, polygon, 255)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=25)
    shadow_strength = random.uniform(0.5, 0.8)
    img_float = img.astype(np.float32)
    shadow = mask.astype(np.float32) / 255.0
    factor = 1.0 - shadow * (1.0 - shadow_strength)
    img_float = img_float * factor[..., None]
    return np.clip(img_float, 0, 255).astype(np.uint8)


def aplicar_ruido(img):
    sigma = random.uniform(5.0, 15.0)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def aplicar_compressao_jpeg(img):
    quality = random.randint(45, 90)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, enc = cv2.imencode(".jpg", img, encode_param)
    if not ok:
        return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec if dec is not None else img


def aplicar_reducao_e_ampliacao(img):
    h, w = img.shape[:2]
    scale = random.uniform(0.5, 0.85)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def aplicar_nitidez(img):
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.2)
    amount = random.uniform(0.8, 1.5)
    out = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def aplicar_vinheta(img):
    h, w = img.shape[:2]
    kernel_x = cv2.getGaussianKernel(w, w * 0.5)
    kernel_y = cv2.getGaussianKernel(h, h * 0.5)
    mask = kernel_y @ kernel_x.T
    mask = mask / mask.max()
    strength = random.uniform(0.5, 0.8)
    vignette = (strength + (1.0 - strength) * mask).astype(np.float32)
    out = img.astype(np.float32) * vignette[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def aplicar_facho_luz(img):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (random.randint(0, w - 1), random.randint(0, h - 1))
    radius = random.randint(int(min(w, h) * 0.15), int(min(w, h) * 0.35))
    cv2.circle(mask, center, radius, 255, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=radius / 2)
    color = np.array(
        [
            random.randint(180, 255),
            random.randint(200, 255),
            random.randint(200, 255),
        ],
        dtype=np.float32,
    )
    strength = random.uniform(0.1, 0.25)
    flare = (mask.astype(np.float32) / 255.0) * strength
    out = img.astype(np.float32) + flare[..., None] * color
    return np.clip(out, 0, 255).astype(np.uint8)


def aplicar_efeitos_camera(imagem):
    if random.random() < CAM_BRIGHT_CONTRAST_PROB:
        imagem = aplicar_brilho_contraste(imagem)
    if random.random() < CAM_WHITE_BALANCE_PROB:
        imagem = aplicar_balanco_branco(imagem)
    if random.random() < CAM_SHADOW_PROB:
        imagem = aplicar_sombra(imagem)
    if random.random() < CAM_NOISE_PROB:
        imagem = aplicar_ruido(imagem)
    if random.random() < CAM_DOWNSCALE_PROB:
        imagem = aplicar_reducao_e_ampliacao(imagem)
    if random.random() < CAM_SHARPEN_PROB:
        imagem = aplicar_nitidez(imagem)
    if random.random() < CAM_VIGNETTE_PROB:
        imagem = aplicar_vinheta(imagem)
    if random.random() < CAM_FLARE_PROB:
        imagem = aplicar_facho_luz(imagem)
    if random.random() < CAM_JPEG_PROB:
        imagem = aplicar_compressao_jpeg(imagem)
    return imagem


def aplicar_efeitos_ambiente(imagem):
    if random.random() < RAIN_PROB:
        imagem = adicionar_chuva_na_regiao(imagem)
    if random.random() < NIGHT_PROB:
        imagem = efeito_noturno(imagem)
    if random.random() < OCCLUSION_PROB:
        imagem = aplicar_oclusao(imagem)
    if random.random() < BLUR_PROB:
        imagem = desfoque_movimento(imagem)
    return imagem


def mascara_dentro_limites(mascara, largura_imagem, altura_imagem, margem):
    ys, xs = np.where(mascara > 0)
    if xs.size == 0 or ys.size == 0:
        return False
    x_min = xs.min()
    x_max = xs.max()
    y_min = ys.min()
    y_max = ys.max()
    return (
        x_min >= margem
        and y_min >= margem
        and x_max <= (largura_imagem - 1 - margem)
        and y_max <= (altura_imagem - 1 - margem)
    )


def aplicar_rotacao_com_tentativas(
    quadro, mascaras, angulo_min, angulo_max, margem, tentativas=6
):
    altura, largura = quadro.shape[:2]
    for _ in range(tentativas):
        angulo = random.uniform(angulo_min, angulo_max)
        novo_quadro, novas_mascaras = rotacionar_rgba_e_mascaras(
            quadro, mascaras, angulo
        )
        if mascara_dentro_limites(novas_mascaras[0], largura, altura, margem):
            return novo_quadro, novas_mascaras
    return quadro, mascaras


def aplicar_perspectiva_com_tentativas(
    quadro, mascaras, max_warp, margem, tentativas=6
):
    altura, largura = quadro.shape[:2]
    for _ in range(tentativas):
        novo_quadro, novas_mascaras = deformar_perspectiva_rgba_e_mascaras(
            quadro, mascaras, max_warp
        )
        if mascara_dentro_limites(novas_mascaras[0], largura, altura, margem):
            return novo_quadro, novas_mascaras
    return quadro, mascaras


def aplicar_curvatura_com_tentativas(
    quadro, mascaras, intensidade_min, intensidade_max, margem, tentativas=6
):
    altura, largura = quadro.shape[:2]
    for _ in range(tentativas):
        intensidade = random.uniform(intensidade_min, intensidade_max)
        if random.random() < 0.5:
            intensidade = -intensidade
        novo_quadro, novas_mascaras = aplicar_deformacao_curvatura(
            quadro, mascaras, intensidade
        )
        if mascara_dentro_limites(novas_mascaras[0], largura, altura, margem):
            return novo_quadro, novas_mascaras
    return quadro, mascaras


def aplicar_lente_com_tentativas(
    quadro, mascaras, k1_min, k1_max, margem, tentativas=6
):
    altura, largura = quadro.shape[:2]
    for _ in range(tentativas):
        k1 = random.uniform(k1_min, k1_max)
        novo_quadro, novas_mascaras = aplicar_distorcao_lente(quadro, mascaras, k1)
        if mascara_dentro_limites(novas_mascaras[0], largura, altura, margem):
            return novo_quadro, novas_mascaras
    return quadro, mascaras


def bbox_mascara(mascara):
    ys, xs = np.where(mascara > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max() + 1)
    y2 = float(ys.max() + 1)
    return x1, y1, x2, y2


# ---------- AJUDANTES DE BBOX ----------
def aplicar_afim_em_pontos(pontos, matriz):
    uns = np.ones((pontos.shape[0], 1), dtype=np.float32)
    pontos_h = np.hstack([pontos.astype(np.float32), uns])
    transformados = (matriz @ pontos_h.T).T
    return transformados


def transformar_bbox(caixa, matriz):
    x1, y1, x2, y2 = caixa
    pontos = np.array(
        [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ],
        dtype=np.float32,
    )
    transformados = aplicar_afim_em_pontos(pontos, matriz)
    xs = transformados[:, 0]
    ys = transformados[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def limitar_bbox(caixa, largura_max, altura_max):
    x1, y1, x2, y2 = caixa
    x1 = max(0.0, min(x1, largura_max))
    x2 = max(0.0, min(x2, largura_max))
    y1 = max(0.0, min(y1, altura_max))
    y2 = max(0.0, min(y2, altura_max))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def bbox_yolo(x1, y1, x2, y2, largura_imagem, altura_imagem):
    w = x2 - x1
    h = y2 - y1
    cx = x1 + (w / 2.0)
    cy = y1 + (h / 2.0)
    return (
        cx / largura_imagem,
        cy / altura_imagem,
        w / largura_imagem,
        h / altura_imagem,
    )


def garantir_pastas_split(diretorio_saida, nome_split):
    pasta_imagens = diretorio_saida / nome_split / "images"
    pasta_rotulos = diretorio_saida / nome_split / "labels"
    os.makedirs(pasta_imagens, exist_ok=True)
    os.makedirs(pasta_rotulos, exist_ok=True)
    return pasta_imagens, pasta_rotulos


def remover_arvore_segura(path):
    if not path.exists():
        return
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            try:
                os.remove(Path(root) / name)
            except Exception:
                pass
        for name in dirs:
            try:
                os.rmdir(Path(root) / name)
            except Exception:
                pass
    try:
        os.rmdir(path)
    except Exception:
        pass


def escrever_metadados_dataset(diretorio_saida, qtd_treino, qtd_validacao, qtd_teste):
    total = qtd_treino + qtd_validacao + qtd_teste
    arquivo_yaml = diretorio_saida / "data.yaml"
    with open(arquivo_yaml, "w", encoding="utf-8") as f:
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        f.write("test: test/images\n\n")
        f.write("nc: 2\n")
        f.write("names: [\"plate\", \"character\"]\n")

    caminho_readme = diretorio_saida / "README.dataset.txt"
    with open(caminho_readme, "w", encoding="utf-8") as f:
        f.write("Dataset sintetico de placas Mercosul.\n")
        f.write(f"Total de imagens: {total}\n")
        f.write(
            f"Split: train {qtd_treino} (70%), valid {qtd_validacao} (20%), "
            f"test {qtd_teste} (10%).\n"
        )
        f.write("Formato de anotacao: YOLO.\n")


def extrair_prefixo_numerico(nome_base):
    prefixo, separador, _ = nome_base.partition("_")
    if separador and prefixo.isdigit():
        return int(prefixo)
    return None


def proximo_indice_inicial(stems_usados):
    maior_indice = 0
    for nome_base in stems_usados:
        indice = extrair_prefixo_numerico(nome_base)
        if indice is not None and indice > maior_indice:
            maior_indice = indice
    return maior_indice + 1


def stem_placa_unico(texto_placa, stems_usados, indice):
    while True:
        nome_base = f"{indice:02d}_{texto_placa}"
        if nome_base not in stems_usados:
            stems_usados.add(nome_base)
            return nome_base, indice + 1
        indice += 1


def gerar_uma_amostra(
    caminho_imagem,
    caminho_rotulo,
    texto_placa,
    argumentos,
    caminhos_fundo,
    templates_placa,
    fonte,
):
    quadro = criar_fundo(argumentos.img_width, argumentos.img_height, caminhos_fundo)
    base_placa = random.choice(templates_placa).copy()
    desenho = ImageDraw.Draw(base_placa)

    mascaras_caracteres = []
    cursor_x = argumentos.text_x

    for caractere in texto_placa:
        imagem_mascara_caractere = Image.new("L", base_placa.size, 0)
        desenho_mascara = ImageDraw.Draw(imagem_mascara_caractere)
        desenho_mascara.text(
            (cursor_x, argumentos.text_y), caractere, font=fonte, fill=255
        )
        mascaras_caracteres.append(np.array(imagem_mascara_caractere))
        desenho.text(
            (cursor_x, argumentos.text_y), caractere, font=fonte, fill=(0, 0, 0, 255)
        )
        caixa_texto = desenho.textbbox((cursor_x, argumentos.text_y), caractere, font=fonte)
        cursor_x += (caixa_texto[2] - caixa_texto[0]) + random.randint(8, 15)

    placa_rgba = np.array(base_placa)
    placa_rgba = cv2.cvtColor(placa_rgba, cv2.COLOR_RGBA2BGRA)
    mascara_placa = placa_rgba[:, :, 3].copy()

    placa_rgb = placa_rgba[:, :, :3].copy()
    if random.random() < PLATE_TINT_PROB:
        placa_rgb = aplicar_tinta_placa(placa_rgb)
    if random.random() < PLATE_DIRT_PROB:
        placa_rgb = adicionar_sujeira_placa(placa_rgb)
    if random.random() < PLATE_SCRATCH_PROB:
        placa_rgb = adicionar_riscos_placa(placa_rgb)
    if random.random() < PLATE_SCREW_PROB:
        placa_rgb = adicionar_parafusos_placa(placa_rgb)
    if random.random() < PLATE_SPECULAR_PROB:
        placa_rgb = adicionar_reflexo_especular(placa_rgb)
    placa_rgba[:, :, :3] = placa_rgb

    mascaras = [mascara_placa] + mascaras_caracteres

    # Escala aleatoria
    escala_aleatoria = random.uniform(argumentos.scale_min, argumentos.scale_max)
    placa_rgba, mascaras = redimensionar_rgba_e_mascaras(
        placa_rgba, mascaras, escala_aleatoria
    )

    # Protecao de tamanho
    largura_maxima_placa = min(
        argumentos.img_width - 40,
        int(argumentos.img_width * argumentos.max_plate_width_ratio),
    )
    altura_maxima_placa = min(
        argumentos.img_height - 40,
        int(argumentos.img_height * argumentos.max_plate_height_ratio),
    )

    altura_placa_tmp, largura_placa_tmp = placa_rgba.shape[:2]
    if largura_placa_tmp > largura_maxima_placa or altura_placa_tmp > altura_maxima_placa:
        escala_ajuste = min(
            largura_maxima_placa / largura_placa_tmp,
            altura_maxima_placa / altura_placa_tmp,
        )
        placa_rgba, mascaras = redimensionar_rgba_e_mascaras(
            placa_rgba, mascaras, escala_ajuste
        )

    # Posicao no quadro
    mascara_placa = mascaras[0]
    mascaras_caracteres = mascaras[1:]
    altura_placa, largura_placa = placa_rgba.shape[:2]
    pos_x_max = argumentos.img_width - largura_placa
    pos_y_max = argumentos.img_height - altura_placa
    margem = int(
        min(argumentos.img_width, argumentos.img_height)
        * argumentos.placement_margin_ratio
    )
    margem_x = min(margem, pos_x_max // 2) if pos_x_max > 0 else 0
    margem_y = min(margem, pos_y_max // 2) if pos_y_max > 0 else 0
    pos_x = random.randint(margem_x, pos_x_max - margem_x) if pos_x_max > 0 else 0
    pos_y = random.randint(margem_y, pos_y_max - margem_y) if pos_y_max > 0 else 0

    mesclar_alpha(quadro, placa_rgba, pos_x, pos_y)

    # Monta mascaras completas no tamanho final
    mascaras_completas = []
    for mascara in [mascara_placa] + mascaras_caracteres:
        mascara_completa = np.zeros(
            (argumentos.img_height, argumentos.img_width), dtype=np.uint8
        )
        mascara_completa[
            pos_y : pos_y + altura_placa, pos_x : pos_x + largura_placa
        ] = mascara
        mascaras_completas.append(mascara_completa)

    # Distorcoes geometricas globais com o fundo (com tentativas)
    margem_warp = max(2, margem)
    quadro, mascaras_completas = aplicar_rotacao_com_tentativas(
        quadro, mascaras_completas, ROTATION_MIN, ROTATION_MAX, margem_warp
    )

    if random.random() < PERSPECTIVE_PROB:
        quadro, mascaras_completas = aplicar_perspectiva_com_tentativas(
            quadro, mascaras_completas, PERSPECTIVE_MAX_WARP, margem_warp
        )

    if random.random() < CURVATURE_PROB:
        quadro, mascaras_completas = aplicar_curvatura_com_tentativas(
            quadro,
            mascaras_completas,
            CURVATURE_STRENGTH_MIN,
            CURVATURE_STRENGTH_MAX,
            margem_warp,
        )

    if random.random() < LENS_DISTORT_PROB:
        quadro, mascaras_completas = aplicar_lente_com_tentativas(
            quadro, mascaras_completas, LENS_K1_MIN, LENS_K1_MAX, margem_warp
        )

    # Efeitos de ambiente e camera
    quadro = aplicar_efeitos_ambiente(quadro)
    quadro = aplicar_efeitos_camera(quadro)

    cv2.imwrite(str(caminho_imagem), quadro)

    with open(caminho_rotulo, "w", encoding="utf-8") as f:
        caixa_placa = bbox_mascara(mascaras_completas[0])
        caixa_limitada = limitar_bbox(
            caixa_placa if caixa_placa is not None else (0.0, 0.0, 0.0, 0.0),
            argumentos.img_width,
            argumentos.img_height,
        )
        if caixa_limitada:
            bx, by, bw, bh = bbox_yolo(
                caixa_limitada[0],
                caixa_limitada[1],
                caixa_limitada[2],
                caixa_limitada[3],
                argumentos.img_width,
                argumentos.img_height,
            )
            f.write(f"0 {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n")

        for mascara_caractere in mascaras_completas[1:]:
            caixa_caractere = bbox_mascara(mascara_caractere)
            if caixa_caractere is None:
                continue
            caixa_limitada = limitar_bbox(
                caixa_caractere,
                argumentos.img_width,
                argumentos.img_height,
            )
            if caixa_limitada is None:
                continue
            bx, by, bw, bh = bbox_yolo(
                caixa_limitada[0],
                caixa_limitada[1],
                caixa_limitada[2],
                caixa_limitada[3],
                argumentos.img_width,
                argumentos.img_height,
            )
            f.write(f"1 {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n")


def analisar_argumentos():
    analisador = argparse.ArgumentParser(
        description="Gera dataset sintetico de placas Mercosul."
    )
    analisador.add_argument("--num-images", type=int, default=DEFAULT_NUM_IMAGES)
    analisador.add_argument("--img-width", type=int, default=DEFAULT_IMG_W)
    analisador.add_argument("--img-height", type=int, default=DEFAULT_IMG_H)
    analisador.add_argument("--plate-template", default=DEFAULT_PLATE_TEMPLATE)
    analisador.add_argument("--font-path", default=DEFAULT_FONT_PATH)
    analisador.add_argument("--font-size", type=int, default=DEFAULT_FONT_SIZE)
    analisador.add_argument("--text-x", type=int, default=DEFAULT_TEXT_X)
    analisador.add_argument("--text-y", type=int, default=DEFAULT_TEXT_Y)
    analisador.add_argument("--scale-min", type=float, default=DEFAULT_SCALE_MIN)
    analisador.add_argument("--scale-max", type=float, default=DEFAULT_SCALE_MAX)
    analisador.add_argument(
        "--max-plate-width-ratio",
        type=float,
        default=DEFAULT_MAX_PLATE_WIDTH_RATIO,
        help="Limite maximo da largura da placa em relacao ao frame (0-1).",
    )
    analisador.add_argument(
        "--max-plate-height-ratio",
        type=float,
        default=DEFAULT_MAX_PLATE_HEIGHT_RATIO,
        help="Limite maximo da altura da placa em relacao ao frame (0-1).",
    )
    analisador.add_argument(
        "--placement-margin-ratio",
        type=float,
        default=DEFAULT_PLACEMENT_MARGIN_RATIO,
        help="Margem minima para posicionar a placa longe das bordas (0-1).",
    )
    analisador.add_argument("--split-train", type=int, default=None)
    analisador.add_argument("--split-valid", type=int, default=None)
    analisador.add_argument("--split-test", type=int, default=None)
    analisador.add_argument(
        "--background-dir",
        default=DEFAULT_BACKGROUND_DIR,
        help="Pasta com imagens de fundo. Se nao existir, usa fundo cinza.",
    )
    analisador.add_argument(
        "--plate-templates-dir",
        default=DEFAULT_PLATE_TEMPLATES_DIR,
        help="Pasta com templates de placa (png/jpg). Se nao existir, usa --plate-template.",
    )
    analisador.add_argument("--output-dir", default="dataset")
    analisador.add_argument("--seed", type=int, default=None)
    analisador.add_argument(
        "--clean",
        action="store_true",
        help="Remove imagens/labels existentes antes de gerar.",
    )
    return analisador.parse_args()


# ================= PRINCIPAL =================
def principal():
    argumentos = analisar_argumentos()

    if argumentos.seed is not None:
        random.seed(argumentos.seed)
        np.random.seed(argumentos.seed)

    if argumentos.scale_min <= 0 or argumentos.scale_max <= 0:
        raise ValueError("scale-min e scale-max devem ser positivos.")
    if argumentos.scale_min > argumentos.scale_max:
        argumentos.scale_min, argumentos.scale_max = (
            argumentos.scale_max,
            argumentos.scale_min,
        )
    if not (0.05 <= argumentos.max_plate_width_ratio <= 1.0):
        raise ValueError("max-plate-width-ratio deve estar entre 0.05 e 1.0")
    if not (0.05 <= argumentos.max_plate_height_ratio <= 1.0):
        raise ValueError("max-plate-height-ratio deve estar entre 0.05 e 1.0")
    if not (0.0 <= argumentos.placement_margin_ratio <= 0.25):
        raise ValueError("placement-margin-ratio deve estar entre 0.0 e 0.25")

    template_placa = Path(argumentos.plate_template)
    caminho_fonte = Path(argumentos.font_path)
    if not caminho_fonte.is_file():
        raise FileNotFoundError(f"Arquivo de fonte nao encontrado: {caminho_fonte}")

    diretorio_saida = Path(argumentos.output_dir)

    contagens_split = [
        argumentos.split_train,
        argumentos.split_valid,
        argumentos.split_test,
    ]
    usar_split = any(qtd is not None for qtd in contagens_split)
    if usar_split and not all(qtd is not None for qtd in contagens_split):
        raise ValueError("split-train, split-valid e split-test devem ser informados.")

    if argumentos.clean:
        if usar_split:
            for nome_split in ("train", "valid", "test"):
                pasta_split = diretorio_saida / nome_split
                remover_arvore_segura(pasta_split)
        else:
            pasta_imagens = diretorio_saida / "images"
            pasta_rotulos = diretorio_saida / "labels"
            remover_arvore_segura(pasta_imagens)
            remover_arvore_segura(pasta_rotulos)

    caminhos_fundo = listar_arquivos_imagem(argumentos.background_dir)
    caminhos_templates = listar_arquivos_imagem(argumentos.plate_templates_dir)

    if not caminhos_templates and not template_placa.is_file():
        raise FileNotFoundError(f"Template de placa nao encontrado: {template_placa}")

    templates_placa = []
    if caminhos_templates:
        for caminho_template in caminhos_templates:
            try:
                templates_placa.append(Image.open(caminho_template).convert("RGBA"))
            except Exception:
                continue
    if not templates_placa:
        if not template_placa.is_file():
            raise FileNotFoundError(f"Template de placa nao encontrado: {template_placa}")
        templates_placa = [Image.open(template_placa).convert("RGBA")]

    fonte = ImageFont.truetype(str(caminho_fonte), argumentos.font_size)

    if usar_split:
        qtd_treino, qtd_validacao, qtd_teste = contagens_split
        os.makedirs(diretorio_saida, exist_ok=True)
        escrever_metadados_dataset(
            diretorio_saida, qtd_treino, qtd_validacao, qtd_teste
        )

        mapa_splits = {
            "train": qtd_treino,
            "valid": qtd_validacao,
            "test": qtd_teste,
        }
        for nome_split, quantidade in mapa_splits.items():
            pasta_imagens, pasta_rotulos = garantir_pastas_split(
                diretorio_saida, nome_split
            )
            stems_imagens = {p.stem for p in pasta_imagens.glob("*.jpg")}
            stems_rotulos = {p.stem for p in pasta_rotulos.glob("*.txt")}
            stems_completos = stems_imagens & stems_rotulos
            stems_usados = stems_imagens | stems_rotulos
            faltantes = max(0, quantidade - len(stems_completos))
            indice_nome = proximo_indice_inicial(stems_usados)
            for _ in range(faltantes):
                texto_placa = gerar_texto_placa()
                nome_base, indice_nome = stem_placa_unico(
                    texto_placa, stems_usados, indice_nome
                )
                caminho_imagem = pasta_imagens / f"{nome_base}.jpg"
                caminho_rotulo = pasta_rotulos / f"{nome_base}.txt"
                gerar_uma_amostra(
                    caminho_imagem,
                    caminho_rotulo,
                    texto_placa,
                    argumentos,
                    caminhos_fundo,
                    templates_placa,
                    fonte,
                )
    else:
        pasta_imagens = diretorio_saida / "images"
        pasta_rotulos = diretorio_saida / "labels"
        os.makedirs(pasta_imagens, exist_ok=True)
        os.makedirs(pasta_rotulos, exist_ok=True)

        stems_imagens = {p.stem for p in pasta_imagens.glob("*.jpg")}
        stems_rotulos = {p.stem for p in pasta_rotulos.glob("*.txt")}
        stems_completos = stems_imagens & stems_rotulos
        stems_usados = stems_imagens | stems_rotulos
        faltantes = max(0, argumentos.num_images - len(stems_completos))
        indice_nome = proximo_indice_inicial(stems_usados)
        for _ in range(faltantes):
            texto_placa = gerar_texto_placa()
            nome_base, indice_nome = stem_placa_unico(
                texto_placa, stems_usados, indice_nome
            )
            caminho_imagem = pasta_imagens / f"{nome_base}.jpg"
            caminho_rotulo = pasta_rotulos / f"{nome_base}.txt"
            gerar_uma_amostra(
                caminho_imagem,
                caminho_rotulo,
                texto_placa,
                argumentos,
                caminhos_fundo,
                templates_placa,
                fonte,
            )

    print("Dataset gerado com sucesso.")


if __name__ == "__main__":
    principal()

