#!/usr/bin/env python3
"""
EyeTracker - AI-Powered Visual Attention & Social Media Viewability Engine
Predicts human visual fixations, renders eye-tracking heatmap overlays,
extracts saccadic scanpaths, and calculates estimated views and CTR across
major social media platforms (YouTube, Instagram, TikTok, X, LinkedIn, etc.).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


# ============================================================================
# Colormaps (Vectorized, Pure NumPy - Zero Matplotlib Dependency)
# ============================================================================

class Colormaps:
    """Fast, smooth, vectorized colormaps for heatmap rendering."""

    @staticmethod
    def turbo(x: np.ndarray) -> np.ndarray:
        """
        Google Turbo colormap polynomial approximation.
        Perceptually uniform, vibrant, high contrast across all light/dark backgrounds.
        Input x: float array in [0, 1]. Returns: uint8 array of shape (*x.shape, 3).
        """
        x = np.clip(x, 0.0, 1.0)
        k_red = [0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943]
        k_green = [0.09140261, 2.19418839, 4.84296658, -14.18503327, 4.27729857, 2.82956604]
        k_blue = [0.10667330, 12.64194608, -60.58204836, 110.36275817, -89.90310912, 27.34824973]

        r = np.polyval(k_red[::-1], x)
        g = np.polyval(k_green[::-1], x)
        b = np.polyval(k_blue[::-1], x)

        rgb = np.stack([r, g, b], axis=-1)
        return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def jet(x: np.ndarray) -> np.ndarray:
        """
        Classic thermal eye-tracking colormap (Blue -> Cyan -> Yellow -> Red).
        """
        x = np.clip(x, 0.0, 1.0)
        r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
        rgb = np.stack([r, g, b], axis=-1)
        return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def inferno(x: np.ndarray) -> np.ndarray:
        """
        Inferno colormap (Black -> Dark Purple -> Red/Orange -> Yellow).
        """
        x = np.clip(x, 0.0, 1.0)
        r = np.clip(x * 1.8, 0.0, 1.0)
        g = np.clip((x - 0.25) * 1.5, 0.0, 1.0) ** 1.4
        b = np.clip(np.sin(x * np.pi) * 0.7 + (x > 0.8) * (x - 0.8) * 1.5, 0.0, 1.0)
        rgb = np.stack([r, g, b], axis=-1)
        return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def hot(x: np.ndarray) -> np.ndarray:
        """
        Hot thermal colormap (Black -> Red -> Yellow -> White).
        """
        x = np.clip(x, 0.0, 1.0)
        r = np.clip(x * 2.7, 0.0, 1.0)
        g = np.clip((x - 0.37) * 2.7, 0.0, 1.0)
        b = np.clip((x - 0.74) * 3.8, 0.0, 1.0)
        rgb = np.stack([r, g, b], axis=-1)
        return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    @classmethod
    def apply(cls, sal_map: np.ndarray, name: str = "turbo") -> np.ndarray:
        """Applies chosen colormap to normalized 2D saliency map (values in [0, 1])."""
        name = name.lower().strip()
        func = getattr(cls, name, cls.turbo)
        return func(sal_map)


# ============================================================================
# Saliency Prediction Engine (Biological & Cognitive Model)
# ============================================================================

def _fast_box_filter_2d(arr: np.ndarray, k: int) -> np.ndarray:
    """Vectorized 2D box filter in pure NumPy using cumulative sums (O(1) per pixel)."""
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    # Horizontal pass
    p1 = np.pad(arr, ((0, 0), (pad + 1, pad)), mode="edge")
    c1 = np.cumsum(p1, axis=1)
    s1 = (c1[:, k:] - c1[:, :-k]) / float(k)
    # Vertical pass
    p0 = np.pad(s1, ((pad + 1, pad), (0, 0)), mode="edge")
    c0 = np.cumsum(p0, axis=0)
    s0 = (c0[k:, :] - c0[:-k, :]) / float(k)
    return s0


def _fast_gaussian_approx_2d(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Fast 3-pass box filter approximating 2D Gaussian blur."""
    if sigma <= 0.5:
        return arr
    k = max(3, int(2 * round(sigma) + 1))
    out = arr
    for _ in range(3):
        out = _fast_box_filter_2d(out, k)
    return out


class SaliencyPredictor:
    """
    Biological visual attention and saliency predictor.
    Combines:
      1. Spectral Residual Analysis (Hou & Zhang, CVPR) in Fourier frequency domain.
      2. Multi-Scale Gradient / Edge Energy (spatial domain details & text density).
      3. Color Opponency (Red-Green & Blue-Yellow retinal channels) & Warmth Bias.
      4. Human Ecological Priors (Center Fixation Bias + Rule-of-Thirds nodal attraction).
      5. Foveal Receptive Field Smoothing.
    """

    def __init__(self, canonical_size: int = 400):
        self.canonical_size = canonical_size

    def _spectral_residual(self, channel: np.ndarray) -> np.ndarray:
        """Computes Fourier spectral residual novelty map for a 2D channel."""
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift) + 1e-8
        phase = np.angle(fshift)
        log_mag = np.log(mag)

        # Local average filter in frequency space (~3.5% of min dimension)
        k = max(3, int(min(channel.shape) * 0.035))
        smooth_log = _fast_box_filter_2d(log_mag, k)

        spectral_res = log_mag - smooth_log
        recon = np.fft.ifftshift(np.exp(spectral_res + 1j * phase))
        saliency = np.abs(np.fft.ifft2(recon)) ** 2
        return saliency

    def _normalize_percentile(self, m: np.ndarray, p_low: float = 12.0, p_high: float = 99.5) -> np.ndarray:
        """Robust percentile clipping and normalization to [0, 1]."""
        low = np.percentile(m, p_low)
        high = np.percentile(m, p_high)
        if high <= low:
            return np.zeros_like(m)
        clipped = np.clip((m - low) / (high - low), 0.0, 1.0)
        return clipped

    def predict(self, img_rgb: Image.Image) -> np.ndarray:
        """
        Takes a PIL RGB image, predicts visual attention saliency map,
        returns normalized 2D float32 array in [0, 1] matching original image resolution.
        """
        orig_w, orig_h = img_rgb.size

        # 1. Downscale to canonical resolution for scale invariance and fast computation
        scale = float(self.canonical_size) / max(orig_h, orig_w)
        sw = max(32, int(round(orig_w * scale)))
        sh = max(32, int(round(orig_h * scale)))
        img_small = img_rgb.resize((sw, sh), Image.Resampling.BILINEAR)

        arr = np.array(img_small, dtype=np.float32) / 255.0
        r_ch, g_ch, b_ch = arr[..., 0], arr[..., 1], arr[..., 2]

        # 2. Decompose into Intensity and Retinal Color Opponent Channels
        intensity = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
        rg_opp = r_ch - g_ch
        by_opp = b_ch - 0.5 * (r_ch + g_ch)

        # 3. Fourier Spectral Residual Novelty across channels
        sal_i = self._spectral_residual(intensity)
        sal_rg = self._spectral_residual(rg_opp)
        sal_by = self._spectral_residual(by_opp)
        sal_spectral = 0.60 * self._normalize_percentile(sal_i) + 0.20 * self._normalize_percentile(sal_rg) + 0.20 * self._normalize_percentile(sal_by)

        # 4. Multi-Scale Gradient / Edge Energy (High-Frequency Details, Text & Faces)
        gy, gx = np.gradient(intensity)
        grad_mag_fine = np.sqrt(gx**2 + gy**2)
        # Coarse gradient
        int_blurred = _fast_gaussian_approx_2d(intensity, 2.5)
        gy_c, gx_c = np.gradient(int_blurred)
        grad_mag_coarse = np.sqrt(gx_c**2 + gy_c**2)
        sal_edges = 0.65 * self._normalize_percentile(grad_mag_fine) + 0.35 * self._normalize_percentile(grad_mag_coarse)

        # 5. Biological Warmth & Skin-Tone Bias
        # Humans are biologically primed to fixate on warm tones (faces, skin, red/yellow warning cues)
        warmth = np.maximum(0.0, r_ch - np.maximum(g_ch, b_ch))
        sal_warmth = self._normalize_percentile(warmth)

        # 6. Natural Human Fixation Prior (Center Bias & Rule of Thirds)
        y_idx, x_idx = np.indices((sh, sw))
        # Center is slightly above vertical midpoint (eyes usually look ~46% from top)
        cx, cy = sw * 0.50, sh * 0.46
        center_gaussian = np.exp(-(((x_idx - cx) ** 2) / (2 * (0.35 * sw) ** 2) + ((y_idx - cy) ** 2) / (2 * (0.32 * sh) ** 2)))

        # Rule of thirds nodes: (1/3, 1/3), (2/3, 1/3), (1/3, 2/3), (2/3, 2/3)
        rot_weight = np.zeros((sh, sw), dtype=np.float32)
        for rx in (0.333, 0.667):
            for ry in (0.333, 0.667):
                rot_weight += np.exp(-(((x_idx - rx * sw) ** 2 + (y_idx - ry * sh) ** 2) / (2 * (0.16 * min(sw, sh)) ** 2)))
        rot_weight /= 4.0
        sal_prior = 0.70 * center_gaussian + 0.30 * rot_weight

        # 7. Saliency Fusion
        combined = (
            0.38 * sal_spectral +
            0.30 * sal_edges +
            0.17 * sal_warmth +
            0.15 * sal_prior
        )

        # 8. Foveal smoothing (human eye fovea covers ~2 degrees of visual field)
        fovea_sigma = max(2.0, min(sw, sh) * 0.035)
        smoothed = _fast_gaussian_approx_2d(combined, fovea_sigma)

        # 9. Non-linear contrast stretching to emphasize focal hot-spots
        stretched = self._normalize_percentile(smoothed, p_low=10.0, p_high=99.0)
        stretched = np.power(stretched, 1.25)

        # 10. Upsample back to original image resolution with bicubic interpolation
        sal_img = Image.fromarray(stretched.astype(np.float32), mode="F")
        sal_orig = sal_img.resize((orig_w, orig_h), Image.Resampling.BICUBIC)
        full_res = np.clip(np.array(sal_orig, dtype=np.float32), 0.0, 1.0)

        # Final re-normalization
        max_val = full_res.max()
        if max_val > 1e-6:
            full_res = full_res / max_val
        return full_res


# ============================================================================
# Saccadic Fixation & Scanpath Extraction
# ============================================================================

@dataclass
class FixationPoint:
    order: int
    x: int
    y: int
    norm_x: float  # [0.0, 1.0]
    norm_y: float  # [0.0, 1.0]
    intensity: float  # [0.0, 1.0]
    attention_share_pct: float  # Percentage of focal attention in surrounding region


class ScanpathExtractor:
    """
    Extracts the sequential saccadic scanpath (where eyes look first, second, third...).
    Uses Gaussian Inhibition of Return (Non-Maximum Suppression) modeling human eye saccades.
    """

    @staticmethod
    def extract(saliency: np.ndarray, num_points: int = 5) -> List[FixationPoint]:
        h, w = saliency.shape
        work_sal = saliency.copy()
        y_grid, x_grid = np.ogrid[:h, :w]
        total_energy = float(saliency.sum()) + 1e-8

        # Inhibition radius (~12% of image dimension)
        inhibit_radius = max(24, int(max(h, w) * 0.12))
        points: List[FixationPoint] = []

        for i in range(1, num_points + 1):
            max_idx = np.unravel_index(np.argmax(work_sal), work_sal.shape)
            py, px = int(max_idx[0]), int(max_idx[1])
            peak_val = float(saliency[py, px])

            if peak_val < 0.05:
                break

            # Calculate local attention share around this fixation (radius = inhibit_radius)
            dist_sq = (x_grid - px) ** 2 + (y_grid - py) ** 2
            local_mask = dist_sq <= (inhibit_radius**2)
            local_sum = float(saliency[local_mask].sum())
            share_pct = (local_sum / total_energy) * 100.0

            pt = FixationPoint(
                order=int(i),
                x=int(px),
                y=int(py),
                norm_x=float(round(px / float(w), 4)),
                norm_y=float(round(py / float(h), 4)),
                intensity=float(round(peak_val, 3)),
                attention_share_pct=float(round(share_pct, 1)),
            )
            points.append(pt)

            # Gaussian inhibition of return (suppress around current fixation)
            inhibition_mask = 1.0 - np.exp(-dist_sq / (2.0 * (inhibit_radius * 0.7) ** 2))
            work_sal *= inhibition_mask

        return points


# ============================================================================
# Heatmap Rendering & Visual Overlay
# ============================================================================

def _get_system_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Loads scalable TTF font or falls back gracefully."""
    font_paths = [
        "/usr/share/fonts/Adwaita/AdwaitaSans-Bold.ttf" if bold else "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    ]
    for p in font_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


class HeatmapRenderer:
    """Renders eye-tracking heatmaps and saccadic scanpaths onto the image."""

    @staticmethod
    def render_overlay(
        orig_img: Image.Image,
        saliency: np.ndarray,
        colormap_name: str = "turbo",
        alpha: float = 0.65,
        draw_colorbar: bool = True,
    ) -> Image.Image:
        """
        Creates smooth alpha-blended heatmap overlay on the original image.
        Low-saliency areas remain naturally transparent so original details are visible.
        """
        w, h = orig_img.size
        # Apply colormap
        heat_rgb = Colormaps.apply(saliency, colormap_name)

        # Alpha channel: transparent where saliency is low (< 0.12), then ramps smoothly
        alpha_map = np.clip((saliency - 0.10) / 0.85, 0.0, 1.0)
        # Apply non-linear curve to alpha for clean, non-murky transparency
        alpha_map = np.power(alpha_map, 0.9) * alpha * 255.0
        alpha_uint8 = alpha_map.astype(np.uint8)

        # Heatmap RGBA
        heat_rgba = np.dstack([heat_rgb, alpha_uint8])
        heat_pil = Image.fromarray(heat_rgba, mode="RGBA")

        # Blend onto original
        base = orig_img.convert("RGBA")
        blended = Image.alpha_composite(base, heat_pil).convert("RGB")

        if draw_colorbar:
            blended = HeatmapRenderer._draw_legend(blended, colormap_name)

        return blended

    @staticmethod
    def _draw_legend(img: Image.Image, colormap_name: str) -> Image.Image:
        """Draws a clean subtle attention scale / colorbar in the corner."""
        w, h = img.size
        leg_w, leg_h = min(160, int(w * 0.18)), 14
        margin_x, margin_y = int(w * 0.03), int(h * 0.04)
        x0, y0 = margin_x, h - margin_y - leg_h

        # Generate mini color gradient
        grad_vals = np.linspace(0.1, 1.0, leg_w).reshape(1, -1)
        grad_vals = np.repeat(grad_vals, leg_h, axis=0)
        grad_rgb = Colormaps.apply(grad_vals, colormap_name)
        grad_pil = Image.fromarray(grad_rgb, mode="RGB")

        # Border and background pill
        pad = 6
        bg_box = (x0 - pad, y0 - 20, x0 + leg_w + pad, y0 + leg_h + pad)
        draw = ImageDraw.Draw(img)
        # Dark translucent backing
        draw.rounded_rectangle(bg_box, radius=6, fill=(18, 20, 26))
        draw.rectangle((x0 - 1, y0 - 1, x0 + leg_w + 1, y0 + leg_h + 1), outline=(100, 110, 130), width=1)
        img.paste(grad_pil, (x0, y0))

        font = _get_system_font(11, bold=True)
        draw.text((x0, y0 - 16), "EYE ATTENTION", fill=(210, 220, 235), font=font)
        draw.text((x0 + leg_w - 24, y0 - 16), "HIGH", fill=(255, 90, 80), font=font)
        return img

    @staticmethod
    def render_scanpath(
        base_img: Image.Image,
        fixations: List[FixationPoint],
        draw_lines: bool = True,
    ) -> Image.Image:
        """
        Draws numbered fixation targets (1st, 2nd, 3rd...) with saccade connecting lines.
        """
        out = base_img.copy()
        draw = ImageDraw.Draw(out)
        w, h = out.size

        badge_r = max(14, int(min(w, h) * 0.026))
        font_size = max(12, int(badge_r * 1.25))
        font = _get_system_font(font_size, bold=True)

        # 1. Draw connecting saccadic path lines
        if draw_lines and len(fixations) > 1:
            for i in range(len(fixations) - 1):
                p1 = (fixations[i].x, fixations[i].y)
                p2 = (fixations[i + 1].x, fixations[i + 1].y)
                # Outer shadow line
                draw.line([p1, p2], fill=(10, 12, 18), width=5)
                # Inner vibrant directional line
                draw.line([p1, p2], fill=(0, 230, 255), width=3)

        # 2. Draw numbered target badges
        for pt in fixations:
            x, y = pt.x, pt.y
            num_str = str(pt.order)

            # Outer glowing halo
            draw.ellipse((x - badge_r - 6, y - badge_r - 6, x + badge_r + 6, y + badge_r + 6), outline=(0, 0, 0), width=4)
            draw.ellipse((x - badge_r - 4, y - badge_r - 4, x + badge_r + 4, y + badge_r + 4), outline=(255, 255, 255), width=2)

            # Badge background color: #1 is vibrant fiery red, #2 orange, #3 yellow, others cyan
            colors = {
                1: (245, 45, 60),
                2: (255, 140, 20),
                3: (250, 210, 30),
                4: (40, 210, 130),
                5: (30, 180, 245),
            }
            fill_col = colors.get(pt.order, (160, 100, 240))
            draw.ellipse((x - badge_r, y - badge_r, x + badge_r, y + badge_r), fill=fill_col, outline=(255, 255, 255), width=2)

            # Number text centered
            # Get text bounding box for exact centering
            try:
                bbox = font.getbbox(num_str)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                tx = x - tw // 2
                ty = y - th // 2 - 1
            except Exception:
                tx, ty = x - 4, y - 6

            text_col = (10, 15, 25) if pt.order in (2, 3) else (255, 255, 255)
            draw.text((tx, ty), num_str, fill=text_col, font=font)

        return out


# ============================================================================
# Social Media Platform Viewability & Approx Views Engine
# ============================================================================

@dataclass
class PlatformCropSpec:
    platform_name: str
    format_label: str
    target_aspect: float  # width / height (e.g. 16/9, 1/1, 9/16)
    aspect_str: str
    # Danger zones defined as normalized boxes [x0, y0, x1, y1] in [0, 1] relative to the platform frame
    danger_zones: List[Tuple[float, float, float, float, str]]
    # Benchmark click-through / engagement baseline
    base_ctr: float
    platform_type: str  # "video_thumb", "feed_post", "vertical_story", "professional"


PLATFORM_SPECS: List[PlatformCropSpec] = [
    PlatformCropSpec(
        platform_name="YouTube",
        format_label="Video Thumbnail",
        target_aspect=16.0 / 9.0,
        aspect_str="16:9",
        danger_zones=[
            (0.78, 0.76, 1.00, 1.00, "Duration Timestamp Badge"),
        ],
        base_ctr=4.2,
        platform_type="video_thumb",
    ),
    PlatformCropSpec(
        platform_name="Instagram",
        format_label="Square Feed Post",
        target_aspect=1.0 / 1.0,
        aspect_str="1:1",
        danger_zones=[],
        base_ctr=5.5,
        platform_type="feed_post",
    ),
    PlatformCropSpec(
        platform_name="Instagram",
        format_label="Portrait Feed Post",
        target_aspect=4.0 / 5.0,
        aspect_str="4:5",
        danger_zones=[],
        base_ctr=6.8,
        platform_type="feed_post",
    ),
    PlatformCropSpec(
        platform_name="TikTok",
        format_label="Full-Screen Video / Cover",
        target_aspect=9.0 / 16.0,
        aspect_str="9:16",
        danger_zones=[
            (0.82, 0.22, 1.00, 0.86, "Action Buttons (Like/Comment/Share)"),
            (0.00, 0.75, 0.85, 1.00, "User Handle & Caption Overlay"),
            (0.00, 0.00, 1.00, 0.12, "Top Navigation Tabs"),
        ],
        base_ctr=7.0,
        platform_type="vertical_story",
    ),
    PlatformCropSpec(
        platform_name="Instagram Reels / Stories",
        format_label="Vertical Story/Reel",
        target_aspect=9.0 / 16.0,
        aspect_str="9:16",
        danger_zones=[
            (0.00, 0.00, 1.00, 0.14, "Top Story Bar & Profile Header"),
            (0.00, 0.84, 1.00, 1.00, "Bottom DM & Engagement Bar"),
        ],
        base_ctr=6.2,
        platform_type="vertical_story",
    ),
    PlatformCropSpec(
        platform_name="X (Twitter)",
        format_label="In-Feed Card",
        target_aspect=16.0 / 9.0,
        aspect_str="16:9",
        danger_zones=[],
        base_ctr=3.8,
        platform_type="feed_post",
    ),
    PlatformCropSpec(
        platform_name="LinkedIn",
        format_label="Feed Post Image",
        target_aspect=1.91 / 1.0,
        aspect_str="1.91:1",
        danger_zones=[],
        base_ctr=3.4,
        platform_type="professional",
    ),
    PlatformCropSpec(
        platform_name="Pinterest",
        format_label="Standard Vertical Pin",
        target_aspect=2.0 / 3.0,
        aspect_str="2:3",
        danger_zones=[
            (0.00, 0.90, 1.00, 1.00, "Bottom Pin Description Overlay"),
        ],
        base_ctr=5.8,
        platform_type="feed_post",
    ),
]


@dataclass
class PlatformAuditResult:
    platform_name: str
    format_label: str
    aspect_str: str
    gaze_retention_pct: float  # % of original image gaze preserved in crop
    safe_zone_pct: float  # % of gaze within safe zone (not occluded by UI)
    ui_occlusion_pct: float  # % of gaze directly blocked by platform UI
    viewability_score: float  # 0 to 100 composite index
    estimated_ctr_pct: float  # Estimated Click-Through Rate %
    estimated_hook_rate_pct: float  # Estimated thumb-stop / hook %
    relative_view_multiplier: float  # Relative efficiency vs image average
    crop_box: Tuple[int, int, int, int]  # (x0, y0, x1, y1) in original image coordinates
    danger_zone_collisions: List[str]  # Warnings regarding specific UI overlaps
    grade: str  # "S", "A+", "A", "B", "C", "D"
    recommendation: str


@dataclass
class EyeTrackingResult:
    image_path: str
    image_width: int
    image_height: int
    aspect_ratio: float
    fixation_points: List[FixationPoint]
    platform_audits: List[PlatformAuditResult]
    clutter_score: float  # 0 (clean, dominant focal subject) to 100 (noisy clutter)
    focal_dominance_score: float  # 0 to 100
    rule_of_thirds_score: float  # 0 to 100
    color_contrast_score: float  # 0 to 100
    best_platform: str
    worst_platform: str
    saliency_map: np.ndarray = field(repr=False)


class SocialMediaAuditor:
    """
    Audits visual attention distribution against social media platform layouts,
    aspect ratios, UI danger zones (timestamps, icons, buttons), and calculates
    approximate views, estimated CTR, and scroll-stopping hook rates.
    """

    @staticmethod
    def audit_image(
        saliency: np.ndarray,
        orig_w: int,
        orig_h: int,
        fixations: List[FixationPoint],
    ) -> Tuple[List[PlatformAuditResult], float, float, float, float]:
        total_saliency = float(saliency.sum()) + 1e-8

        # Global Image Aesthetics & Attention Metrics
        # 1. Focal Dominance: Ratio of top fixation energy to median background
        top_sal = float(saliency.max())
        mean_sal = float(saliency.mean())
        focal_ratio = min(10.0, top_sal / max(mean_sal, 1e-6))
        focal_dominance_score = float(round(min(100.0, focal_ratio * 14.0), 1))

        # 2. Clutter Score: High entropy / noisy energy across whole frame
        p90 = float(np.percentile(saliency, 90))
        p10 = float(np.percentile(saliency, 10))
        clutter_score = float(round(float(np.clip((1.0 - (p90 - p10)) * 100.0, 5.0, 95.0)), 1))

        # 3. Rule of Thirds Score: Distance of primary fixation (#1) to closest RoT intersection
        if fixations:
            f1 = fixations[0]
            rot_targets = [(orig_w * rx, orig_h * ry) for rx in (0.333, 0.667) for ry in (0.333, 0.667)]
            min_dist = min(math.hypot(f1.x - tx, f1.y - ty) for tx, ty in rot_targets)
            diag = math.hypot(orig_w, orig_h)
            rot_score = float(round(max(10.0, min(100.0, (1.0 - (min_dist / (0.45 * diag))) * 100.0)), 1))
        else:
            rot_score = 50.0

        # 4. Color Contrast / Dynamic Range Score
        color_contrast_score = float(round(min(100.0, float((p90 / max(top_sal, 1e-6)) * 85.0 + 15.0)), 1))

        audits: List[PlatformAuditResult] = []

        for spec in PLATFORM_SPECS:
            # Compute center crop box for target aspect ratio
            crop_box = SocialMediaAuditor._compute_crop_box(orig_w, orig_h, spec.target_aspect)
            cx0, cy0, cx1, cy1 = crop_box
            cw = cx1 - cx0
            ch = cy1 - cy0

            crop_sal = saliency[cy0:cy1, cx0:cx1]
            crop_sum = float(crop_sal.sum())
            gaze_retention = (crop_sum / total_saliency) * 100.0

            # Evaluate UI Danger Zones inside the crop
            occluded_sum = 0.0
            collisions: List[str] = []

            for dz_x0, dz_y0, dz_x1, dz_y1, dz_label in spec.danger_zones:
                bx0 = int(dz_x0 * cw)
                by0 = int(dz_y0 * ch)
                bx1 = int(dz_x1 * cw)
                by1 = int(dz_y1 * ch)

                dz_sal = crop_sal[by0:by1, bx0:bx1]
                dz_energy = float(dz_sal.sum())
                dz_pct = (dz_energy / max(crop_sum, 1e-6)) * 100.0

                if dz_pct > 4.0:  # Significant gaze trapped under UI
                    occluded_sum += dz_energy
                    collisions.append(f"{dz_label} blocks {dz_pct:.1f}% of gaze attention")

            ui_occlusion_pct = min(100.0, (occluded_sum / max(crop_sum, 1e-6)) * 100.0)
            safe_zone_pct = max(0.0, 100.0 - ui_occlusion_pct)

            # Viewability Index (0 - 100)
            retention_factor = gaze_retention / 100.0
            safe_factor = safe_zone_pct / 100.0
            clarity_factor = (100.0 - clutter_score * 0.5) / 100.0

            viewability = (
                retention_factor * 40.0 +
                safe_factor * 30.0 +
                clarity_factor * 20.0 +
                (rot_score / 100.0) * 10.0
            )
            viewability = round(max(10.0, min(99.0, viewability)), 1)

            # Platform-Specific Estimated CTR & Hook Rate
            # Scaled non-linearly with viewability index
            ctr_scale = (viewability / 100.0) ** 1.35
            estimated_ctr = round(spec.base_ctr * (0.55 + 0.95 * ctr_scale), 2)

            hook_rate = round(30.0 + 65.0 * (viewability / 100.0), 1)

            # Letter grade
            if viewability >= 90:
                grade = "S"
            elif viewability >= 82:
                grade = "A+"
            elif viewability >= 74:
                grade = "A"
            elif viewability >= 65:
                grade = "B"
            elif viewability >= 52:
                grade = "C"
            else:
                grade = "D"

            # Recommendations
            recommendations: List[str] = []
            if gaze_retention < 75.0:
                recommendations.append(f"Crop cuts off {100.0 - gaze_retention:.0f}% of attention. Reframe subject closer to center.")
            if collisions:
                recommendations.append("Relocate focal points away from UI overlay danger zones.")
            if clutter_score > 60.0:
                recommendations.append("High visual clutter; simplify background to boost thumb-stop rate.")
            if not recommendations:
                recommendations.append("Optimal framing and visual gaze alignment!")

            audits.append(
                PlatformAuditResult(
                    platform_name=spec.platform_name,
                    format_label=spec.format_label,
                    aspect_str=spec.aspect_str,
                    gaze_retention_pct=round(gaze_retention, 1),
                    safe_zone_pct=round(safe_zone_pct, 1),
                    ui_occlusion_pct=round(ui_occlusion_pct, 1),
                    viewability_score=viewability,
                    estimated_ctr_pct=estimated_ctr,
                    estimated_hook_rate_pct=hook_rate,
                    relative_view_multiplier=1.0,  # Will normalize below
                    crop_box=crop_box,
                    danger_zone_collisions=collisions,
                    grade=grade,
                    recommendation=" ".join(recommendations),
                )
            )

        # Compute relative view multiplier (how this image performs on platform X vs average)
        avg_score = sum(a.viewability_score for a in audits) / len(audits)
        for a in audits:
            a.relative_view_multiplier = round(a.viewability_score / max(avg_score, 1e-6), 2)

        return audits, clutter_score, focal_dominance_score, rot_score, color_contrast_score

    @staticmethod
    def _compute_crop_box(w: int, h: int, target_aspect: float) -> Tuple[int, int, int, int]:
        """Calculates standard center crop coordinates (x0, y0, x1, y1) for an aspect ratio."""
        curr_aspect = float(w) / float(h)
        if abs(curr_aspect - target_aspect) < 0.02:
            return (0, 0, w, h)

        if curr_aspect > target_aspect:
            # Image is wider than target: crop left & right
            new_w = int(round(h * target_aspect))
            x0 = max(0, (w - new_w) // 2)
            x1 = x0 + new_w
            return (x0, 0, x1, h)
        else:
            # Image is taller than target: crop top & bottom
            new_h = int(round(w / target_aspect))
            y0 = max(0, (h - new_h) // 2)
            y1 = y0 + new_h
            return (0, y0, w, y1)


# ============================================================================
# Multi-Platform Visual Simulation Preview Card
# ============================================================================

class SocialPreviewRenderer:
    """
    Renders a side-by-side composite comparison board showing:
      1. Original Image with Heatmap Overlay & Scanpath.
      2. Simulated YouTube Thumbnail (with bottom-right duration badge).
      3. Simulated Instagram 1:1 Feed Crop (with retention badge).
      4. Simulated TikTok 9:16 Vertical Video (with right buttons & caption zone).
    """

    @staticmethod
    def render_audit_card(
        orig_img: Image.Image,
        heatmap_overlay: Image.Image,
        scanpath_img: Image.Image,
        audits: List[PlatformAuditResult],
    ) -> Image.Image:
        card_w, card_h = 1600, 980
        canvas = Image.new("RGB", (card_w, card_h), (16, 18, 26))
        draw = ImageDraw.Draw(canvas)

        font_header = _get_system_font(28, bold=True)
        font_sub = _get_system_font(15, bold=False)
        font_card_title = _get_system_font(18, bold=True)
        font_badge = _get_system_font(13, bold=True)
        font_stat = _get_system_font(12, bold=False)

        # Header Title
        draw.text((45, 30), "AI EYE-TRACKING & SOCIAL MEDIA VIEWABILITY AUDIT", fill=(255, 255, 255), font=font_header)
        draw.text((45, 68), "Human visual attention prediction & cross-platform safe-zone compliance", fill=(140, 155, 180), font=font_sub)

        # -------------------------------------------------------------
        # Top Row: 3 main views (Original, Heatmap Overlay, Fixation Scanpath)
        # -------------------------------------------------------------
        cell_y = 105
        cell_w, cell_h = 475, 275

        # View 1: Original
        SocialPreviewRenderer._render_view_cell(
            canvas, orig_img, (45, cell_y, 45 + cell_w, cell_y + cell_h),
            "1. ORIGINAL CREATIVE", "(Source input frame)", font_card_title, font_stat
        )

        # View 2: Heatmap Overlay
        SocialPreviewRenderer._render_view_cell(
            canvas, heatmap_overlay, (560, cell_y, 560 + cell_w, cell_y + cell_h),
            "2. ATTENTION HEATMAP", "Where human eyes look first (warm = hot)", font_card_title, font_stat
        )

        # View 3: Fixation Scanpath
        SocialPreviewRenderer._render_view_cell(
            canvas, scanpath_img, (1075, cell_y, 1075 + cell_w, cell_y + cell_h),
            "3. SACCADIC SCANPATH", "Fixation sequence: 1st -> 2nd -> 3rd hook", font_card_title, font_stat
        )

        # -------------------------------------------------------------
        # Bottom Row: 3 Platform Simulations (YouTube, Instagram 1:1, TikTok 9:16)
        # -------------------------------------------------------------
        bot_y = 440
        bot_h = 480

        # Find audit specs
        yt_audit = next((a for a in audits if a.platform_name == "YouTube"), audits[0])
        ig_audit = next((a for a in audits if a.format_label == "Square Feed Post"), audits[1])
        tt_audit = next((a for a in audits if a.platform_name == "TikTok"), audits[3])

        # Platform 1: YouTube 16:9 Thumbnail
        yt_cell_x0, yt_cell_x1 = 45, 520
        SocialPreviewRenderer._render_youtube_sim(
            canvas, orig_img, yt_audit, (yt_cell_x0, bot_y, yt_cell_x1, bot_y + bot_h),
            font_card_title, font_badge, font_stat
        )

        # Platform 2: Instagram 1:1 Square
        ig_cell_x0, ig_cell_x1 = 560, 1035
        SocialPreviewRenderer._render_instagram_sim(
            canvas, orig_img, ig_audit, (ig_cell_x0, bot_y, ig_cell_x1, bot_y + bot_h),
            font_card_title, font_badge, font_stat
        )

        # Platform 3: TikTok 9:16 Vertical Video Cover
        tt_cell_x0, tt_cell_x1 = 1075, 1550
        SocialPreviewRenderer._render_tiktok_sim(
            canvas, orig_img, tt_audit, (tt_cell_x0, bot_y, tt_cell_x1, bot_y + bot_h),
            font_card_title, font_badge, font_stat
        )

        return canvas

    @staticmethod
    def _render_view_cell(
        canvas: Image.Image,
        content_img: Image.Image,
        box: Tuple[int, int, int, int],
        title: str,
        subtitle: str,
        font_title: Any,
        font_sub: Any,
    ):
        x0, y0, x1, y1 = box
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle(box, radius=8, fill=(24, 28, 40), outline=(45, 50, 70), width=1)
        draw.text((x0 + 15, y0 + 12), title, fill=(240, 245, 255), font=font_title)
        draw.text((x0 + 15, y0 + 36), subtitle, fill=(130, 140, 165), font=font_sub)

        # Inner image thumbnail
        img_x0, img_y0 = x0 + 15, y0 + 58
        avail_w, avail_h = (x1 - x0) - 30, (y1 - y0) - 72

        # Scale to fit while preserving aspect ratio
        thumb = content_img.copy()
        thumb.thumbnail((avail_w, avail_h), Image.Resampling.BILINEAR)
        tw, th = thumb.size
        # Center horizontally and vertically inside available area
        px = img_x0 + (avail_w - tw) // 2
        py = img_y0 + (avail_h - th) // 2
        canvas.paste(thumb, (px, py))
        draw.rectangle((px, py, px + tw, py + th), outline=(60, 70, 95), width=1)

    @staticmethod
    def _render_youtube_sim(
        canvas: Image.Image,
        orig_img: Image.Image,
        audit: PlatformAuditResult,
        box: Tuple[int, int, int, int],
        font_title: Any,
        font_badge: Any,
        font_stat: Any,
    ):
        x0, y0, x1, y1 = box
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle(box, radius=10, fill=(22, 26, 38), outline=(45, 50, 70), width=1)

        # Card Title & Badge
        draw.text((x0 + 18, y0 + 15), "YOUTUBE THUMBNAIL (16:9)", fill=(255, 60, 60), font=font_title)
        # Grade Pill
        draw.rounded_rectangle((x1 - 65, y0 + 14, x1 - 18, y0 + 38), radius=6, fill=(255, 45, 65))
        draw.text((x1 - 50, y0 + 17), audit.grade, fill=(255, 255, 255), font=font_badge)

        # Crop original to 16:9
        crop_img = orig_img.crop(audit.crop_box)
        preview_w = (x1 - x0) - 36
        preview_h = int(preview_w * (9.0 / 16.0))
        thumb = crop_img.resize((preview_w, preview_h), Image.Resampling.BILINEAR)

        # Draw YouTube Timestamp Overlay onto the thumbnail
        # Duration badge at bottom-right: "14:28"
        t_draw = ImageDraw.Draw(thumb)
        bw, bh = int(preview_w * 0.22), int(preview_h * 0.18)
        bx0, by0 = preview_w - bw - 8, preview_h - bh - 8
        t_draw.rounded_rectangle((bx0, by0, bx0 + bw, by0 + bh), radius=4, fill=(10, 10, 10), outline=(255, 40, 40) if audit.ui_occlusion_pct > 0 else None, width=2)
        t_font = _get_system_font(max(10, bh // 2), bold=True)
        t_draw.text((bx0 + 8, by0 + 4), "14:28", fill=(255, 255, 255), font=t_font)

        py = y0 + 50
        canvas.paste(thumb, (x0 + 18, py))
        draw.rectangle((x0 + 18, py, x0 + 18 + preview_w, py + preview_h), outline=(70, 80, 105), width=1)

        # Metrics stats below
        sy = py + preview_h + 20
        SocialPreviewRenderer._draw_stat_row(draw, x0 + 18, sy, "Estimated CTR:", f"{audit.estimated_ctr_pct}%", (0, 240, 180), font_stat)
        SocialPreviewRenderer._draw_stat_row(draw, x0 + 18, sy + 24, "Viewability Score:", f"{audit.viewability_score}/100", (255, 220, 40), font_stat)
        SocialPreviewRenderer._draw_stat_row(draw, x0 + 18, sy + 48, "Gaze Retention:", f"{audit.gaze_retention_pct}%", (240, 245, 255), font_stat)
        SocialPreviewRenderer._draw_stat_row(draw, x0 + 18, sy + 72, "Duration Occlusion:", f"{audit.ui_occlusion_pct}%", (255, 80, 80) if audit.ui_occlusion_pct > 0 else (120, 220, 120), font_stat)

        # Advisory / Warning
        adv_y = sy + 105
        warn_text = audit.danger_zone_collisions[0] if audit.danger_zone_collisions else "Safe from duration badge collision."
        draw.text((x0 + 18, adv_y), f"Advice: {warn_text[:55]}", fill=(210, 180, 120) if audit.danger_zone_collisions else (120, 190, 140), font=font_stat)

    @staticmethod
    def _render_instagram_sim(
        canvas: Image.Image,
        orig_img: Image.Image,
        audit: PlatformAuditResult,
        box: Tuple[int, int, int, int],
        font_title: Any,
        font_badge: Any,
        font_stat: Any,
    ):
        x0, y0, x1, y1 = box
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle(box, radius=10, fill=(22, 26, 38), outline=(45, 50, 70), width=1)

        draw.text((x0 + 18, y0 + 15), "INSTAGRAM FEED (1:1 SQUARE)", fill=(225, 75, 170), font=font_title)
        draw.rounded_rectangle((x1 - 65, y0 + 14, x1 - 18, y0 + 38), radius=6, fill=(220, 50, 140))
        draw.text((x1 - 50, y0 + 17), audit.grade, fill=(255, 255, 255), font=font_badge)

        # Crop to 1:1
        crop_img = orig_img.crop(audit.crop_box)
        side = min((x1 - x0) - 36, 260)
        thumb = crop_img.resize((side, side), Image.Resampling.BILINEAR)

        py = y0 + 50
        px = x0 + ((x1 - x0) - side) // 2
        canvas.paste(thumb, (px, py))
        draw.rectangle((px, py, px + side, py + side), outline=(70, 80, 105), width=1)

        sy = py + side + 15
        SocialPreviewRenderer._draw_stat_row(draw, x0 + 18, sy, "Thumb-Stop Rate:", f"{audit.estimated_hook_rate_pct}%", (0, 240, 180), font_stat)
        SocialPreviewRenderer._draw_stat_row(draw, x0 + 18, sy + 24, "Viewability Score:", f"{audit.viewability_score}/100", (255, 220, 40), font_stat)
        SocialPreviewRenderer._draw_stat_row(draw, x0 + 18, sy + 48, "1:1 Gaze Retention:", f"{audit.gaze_retention_pct}%", (240, 245, 255), font_stat)
        SocialPreviewRenderer._draw_stat_row(draw, x0 + 18, sy + 72, "Relative Multiplier:", f"{audit.relative_view_multiplier}x vs Avg", (180, 210, 255), font_stat)

        adv_y = sy + 105
        draw.text((x0 + 18, adv_y), f"Advice: {audit.recommendation[:55]}", fill=(160, 210, 170), font=font_stat)

    @staticmethod
    def _render_tiktok_sim(
        canvas: Image.Image,
        orig_img: Image.Image,
        audit: PlatformAuditResult,
        box: Tuple[int, int, int, int],
        font_title: Any,
        font_badge: Any,
        font_stat: Any,
    ):
        x0, y0, x1, y1 = box
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle(box, radius=10, fill=(22, 26, 38), outline=(45, 50, 70), width=1)

        draw.text((x0 + 18, y0 + 15), "TIKTOK / REELS (9:16 VERTICAL)", fill=(0, 240, 220), font=font_title)
        draw.rounded_rectangle((x1 - 65, y0 + 14, x1 - 18, y0 + 38), radius=6, fill=(0, 190, 175))
        draw.text((x1 - 50, y0 + 17), audit.grade, fill=(10, 20, 25), font=font_badge)

        crop_img = orig_img.crop(audit.crop_box)
        preview_h = 260
        preview_w = int(preview_h * (9.0 / 16.0))
        thumb = crop_img.resize((preview_w, preview_h), Image.Resampling.BILINEAR)

        # Draw TikTok UI Overlays: right side icons (Heart, Comment, Share) and bottom caption bar
        t_draw = ImageDraw.Draw(thumb)
        # Right sidebar icons
        rx = preview_w - 18
        for iy, sym in [(100, "♥"), (135, "💬"), (170, "↗")]:
            t_draw.ellipse((rx - 7, iy - 7, rx + 7, iy + 7), fill=(20, 20, 25))
            t_draw.text((rx - 4, iy - 6), sym, fill=(255, 255, 255), font=_get_system_font(8))

        # Bottom caption overlay
        t_draw.rectangle((0, preview_h - 40, preview_w, preview_h), fill=(10, 10, 15))
        t_draw.text((8, preview_h - 32), "@creator", fill=(255, 255, 255), font=_get_system_font(9, bold=True))
        t_draw.text((8, preview_h - 18), "🎵 Trending audio", fill=(200, 200, 220), font=_get_system_font(7))

        py = y0 + 50
        px = x0 + ((x1 - x0) - preview_w) // 2
        canvas.paste(thumb, (px, py))
        draw.rectangle((px, py, px + preview_w, py + preview_h), outline=(70, 80, 105), width=1)

        sy = py + preview_h + 15
        SocialPreviewRenderer._draw_stat_row(draw, x0 + 18, sy, "3-Sec Hook Rate:", f"{audit.estimated_hook_rate_pct}%", (0, 240, 180), font_stat)
        SocialPreviewRenderer._draw_stat_row(draw, x0 + 18, sy + 24, "Viewability Score:", f"{audit.viewability_score}/100", (255, 220, 40), font_stat)
        SocialPreviewRenderer._draw_stat_row(draw, x0 + 18, sy + 48, "Vertical Retention:", f"{audit.gaze_retention_pct}%", (240, 245, 255), font_stat)
        SocialPreviewRenderer._draw_stat_row(draw, x0 + 18, sy + 72, "UI Safe Compliance:", f"{audit.safe_zone_pct}%", (240, 245, 255), font_stat)

        adv_y = sy + 105
        draw.text((x0 + 18, adv_y), f"Advice: {audit.recommendation[:55]}", fill=(160, 210, 170), font=font_stat)

    @staticmethod
    def _draw_stat_row(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, value: str, val_col: Tuple[int, int, int], font: Any):
        draw.text((x, y), label, fill=(165, 175, 195), font=font)
        draw.text((x + 160, y), value, fill=val_col, font=font)


# ============================================================================
# High-Level EyeTracker Facade & API
# ============================================================================

def _json_serializer(obj):
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


class EyeTracker:
    """
    Main EyeTracker facade for running visual attention prediction,
    generating heatmap overlays, and computing social media view metrics.
    """

    def __init__(self, canonical_size: int = 400):
        self.predictor = SaliencyPredictor(canonical_size=canonical_size)

    def analyze(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        num_fixation_points: int = 5,
    ) -> EyeTrackingResult:
        """
        Analyzes an input image (file path, PIL Image, or NumPy array),
        returns a comprehensive EyeTrackingResult.
        """
        if isinstance(image_input, (str, Path)):
            path_str = str(image_input)
            pil_img = Image.open(path_str).convert("RGB")
        elif isinstance(image_input, Image.Image):
            path_str = "<PIL.Image>"
            pil_img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            path_str = "<numpy.ndarray>"
            if image_input.dtype != np.uint8:
                image_input = np.clip(image_input * 255.0, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(image_input).convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        w, h = pil_img.size

        # 1. Compute biological visual saliency map
        saliency = self.predictor.predict(pil_img)

        # 2. Extract saccadic scanpath fixation points
        fixations = ScanpathExtractor.extract(saliency, num_points=num_fixation_points)

        # 3. Audit against social media platforms
        audits, clutter, focal, rot, contrast = SocialMediaAuditor.audit_image(saliency, w, h, fixations)

        # Determine best and worst platforms
        sorted_audits = sorted(audits, key=lambda a: a.viewability_score, reverse=True)
        best_plat = f"{sorted_audits[0].platform_name} ({sorted_audits[0].format_label})"
        worst_plat = f"{sorted_audits[-1].platform_name} ({sorted_audits[-1].format_label})"

        return EyeTrackingResult(
            image_path=path_str,
            image_width=w,
            image_height=h,
            aspect_ratio=round(float(w) / float(h), 3),
            fixation_points=fixations,
            platform_audits=audits,
            clutter_score=clutter,
            focal_dominance_score=focal,
            rule_of_thirds_score=rot,
            color_contrast_score=contrast,
            best_platform=best_plat,
            worst_platform=worst_plat,
            saliency_map=saliency,
        )

    def render_all_outputs(
        self,
        orig_img: Image.Image,
        result: EyeTrackingResult,
        output_dir: Union[str, Path],
        colormap: str = "turbo",
        alpha: float = 0.65,
        base_name: str = "eyetracker",
    ) -> Dict[str, str]:
        """
        Renders and saves all visualization artifacts:
          - Heatmap overlay image
          - Scanpath fixation order diagram
          - Raw grayscale saliency map
          - Comprehensive multi-platform audit grid card
          - JSON metrics report
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        saved_files: Dict[str, str] = {}

        # 1. Heatmap Overlay
        heatmap_overlay = HeatmapRenderer.render_overlay(
            orig_img, result.saliency_map, colormap_name=colormap, alpha=alpha, draw_colorbar=True
        )
        p_heat = str(out_path / f"{base_name}_heatmap_overlay.png")
        heatmap_overlay.save(p_heat, quality=95)
        saved_files["heatmap_overlay"] = p_heat

        # 2. Scanpath diagram
        scanpath_img = HeatmapRenderer.render_scanpath(orig_img, result.fixation_points)
        p_scan = str(out_path / f"{base_name}_scanpath.png")
        scanpath_img.save(p_scan, quality=95)
        saved_files["scanpath"] = p_scan

        # 3. Raw Grayscale Saliency Map
        raw_sal_uint8 = (result.saliency_map * 255.0).astype(np.uint8)
        raw_pil = Image.fromarray(raw_sal_uint8, mode="L")
        p_raw = str(out_path / f"{base_name}_saliency_raw.png")
        raw_pil.save(p_raw)
        saved_files["saliency_raw"] = p_raw

        # 4. Multi-Platform Audit Collage Card
        audit_card = SocialPreviewRenderer.render_audit_card(
            orig_img, heatmap_overlay, scanpath_img, result.platform_audits
        )
        p_card = str(out_path / f"{base_name}_social_audit_grid.png")
        audit_card.save(p_card, quality=95)
        saved_files["social_audit_grid"] = p_card

        # 5. JSON metrics report
        report_data = {
            "image_path": result.image_path,
            "dimensions": {"width": result.image_width, "height": result.image_height, "aspect_ratio": result.aspect_ratio},
            "scores": {
                "clutter_score": result.clutter_score,
                "focal_dominance_score": result.focal_dominance_score,
                "rule_of_thirds_score": result.rule_of_thirds_score,
                "color_contrast_score": result.color_contrast_score,
            },
            "summary": {
                "best_performing_platform": result.best_platform,
                "lowest_performing_platform": result.worst_platform,
            },
            "fixation_scanpath": [asdict(fp) for fp in result.fixation_points],
            "social_platforms": [asdict(pa) for pa in result.platform_audits],
        }
        p_json = str(out_path / f"{base_name}_report.json")
        with open(p_json, "w") as f:
            json.dump(report_data, f, indent=2, default=_json_serializer)
        saved_files["json_report"] = p_json

        return saved_files


# ============================================================================
# Terminal Reporting (Rich Tables / Plain Text Fallback)
# ============================================================================

def print_audit_report(result: EyeTrackingResult):
    """Outputs a clean, professional summary table of all social media views."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()

        # Overview Panel
        summary_text = (
            f"[bold cyan]Input Dimensions:[/bold cyan] {result.image_width}x{result.image_height} (Aspect: {result.aspect_ratio})\n"
            f"[bold cyan]Focal Hook Dominance:[/bold cyan] {result.focal_dominance_score}/100 | "
            f"[bold cyan]Clutter Level:[/bold cyan] {result.clutter_score}/100\n"
            f"[bold green]Best Platform Fit:[/bold green] [bold]{result.best_platform}[/bold]\n"
            f"[bold red]Highest Risk Fit:[/bold red] [bold]{result.worst_platform}[/bold]"
        )
        console.print(Panel(summary_text, title="[bold white]EyeTracker Visual Attention Audit[/bold white]", border_style="cyan"))

        # Fixations Table
        fix_table = Table(title="Human Fixation Saccade Sequence (Order of Attention)")
        fix_table.add_column("Order", justify="center", style="bold yellow")
        fix_table.add_column("Coordinates (X, Y)", justify="center")
        fix_table.add_column("Attention Energy", justify="center", style="cyan")
        fix_table.add_column("Local Attention Share", justify="center", style="green")

        for fp in result.fixation_points:
            fix_table.add_row(
                f"#{fp.order}",
                f"({fp.x}, {fp.y})",
                f"{fp.intensity * 100:.0f}%",
                f"{fp.attention_share_pct:.1f}%",
            )
        console.print(fix_table)

        # Platform Views & Performance Table
        table = Table(title="Social Media Platform Viewability & Approx Views Analysis")
        table.add_column("Platform / Placement", style="bold white")
        table.add_column("Aspect", justify="center", style="dim")
        table.add_column("Gaze Kept", justify="right", style="cyan")
        table.add_column("UI Safe", justify="right")
        table.add_column("Est. CTR", justify="right", style="bold green")
        table.add_column("Hook Stop %", justify="right", style="magenta")
        table.add_column("Score", justify="right", style="bold yellow")
        table.add_column("Grade", justify="center")
        table.add_column("View Multiplier", justify="center", style="blue")
        table.add_column("Notes / Warnings")

        for a in result.platform_audits:
            safe_style = "green" if a.safe_zone_pct >= 95 else ("yellow" if a.safe_zone_pct >= 85 else "bold red")
            grade_style = "bold green" if a.grade in ("S", "A+", "A") else ("yellow" if a.grade == "B" else "red")
            notes = a.danger_zone_collisions[0] if a.danger_zone_collisions else "Optimal"

            table.add_row(
                f"{a.platform_name} - {a.format_label}",
                a.aspect_str,
                f"{a.gaze_retention_pct:.1f}%",
                f"[{safe_style}]{a.safe_zone_pct:.1f}%[/{safe_style}]",
                f"{a.estimated_ctr_pct:.2f}%",
                f"{a.estimated_hook_rate_pct:.1f}%",
                f"{a.viewability_score:.1f}",
                f"[{grade_style}]{a.grade}[/{grade_style}]",
                f"{a.relative_view_multiplier:.2f}x",
                f"[dim]{notes[:38]}[/dim]",
            )
        console.print(table)

    except ImportError:
        # Fallback to plain text formatting
        print("\n" + "=" * 80)
        print("EYETRACKER - VISUAL ATTENTION & SOCIAL MEDIA VIEWABILITY REPORT")
        print("=" * 80)
        print(f"Dimensions: {result.image_width}x{result.image_height} (Aspect: {result.aspect_ratio})")
        print(f"Clutter: {result.clutter_score}/100 | Focal Dominance: {result.focal_dominance_score}/100")
        print(f"Best Platform: {result.best_platform}")
        print(f"Worst Platform: {result.worst_platform}\n")

        print("FIXATION SCANPATH:")
        for fp in result.fixation_points:
            print(f"  Fixation #{fp.order}: ({fp.x}, {fp.y}) - Attention Share: {fp.attention_share_pct:.1f}%")

        print("\nSOCIAL MEDIA PLATFORM AUDIT:")
        print(f"{'Platform':<32} {'Aspect':<8} {'Gaze %':<8} {'Safe %':<8} {'Est CTR':<10} {'Hook %':<8} {'Score':<8} {'Grade'}")
        print("-" * 92)
        for a in result.platform_audits:
            name = f"{a.platform_name} ({a.format_label})"[:30]
            print(f"{name:<32} {a.aspect_str:<8} {a.gaze_retention_pct:<8.1f} {a.safe_zone_pct:<8.1f} {a.estimated_ctr_pct:<9.2f}% {a.estimated_hook_rate_pct:<8.1f} {a.viewability_score:<8.1f} {a.grade}")
        print("=" * 80)


# ============================================================================
# CLI Command-Line Interface
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict where humans look at an image, generate eye-tracking heatmaps, and calculate approx social media views.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Path to the input image. If omitted and --demo is set, a sample image is generated.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./eyetracker_output",
        help="Directory to save generated heatmaps, scanpaths, audit grids, and reports.",
    )
    parser.add_argument(
        "--colormap", "-c",
        type=str,
        choices=["turbo", "jet", "inferno", "hot"],
        default="turbo",
        help="Heatmap colormap palette.",
    )
    parser.add_argument(
        "--alpha", "-a",
        type=float,
        default=0.65,
        help="Heatmap overlay opacity between 0.1 and 1.0.",
    )
    parser.add_argument(
        "--points", "-p",
        type=int,
        default=5,
        help="Number of fixation scanpath points to extract (1st, 2nd, 3rd hook...).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate and audit a realistic sample YouTube thumbnail demo creative.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON output to stdout.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress terminal tables.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    image_path = args.image

    # If no image provided and demo requested or no args given, generate demo
    if image_path is None:
        if args.demo or len(sys.argv) == 1:
            from demo_assets import generate_youtube_thumbnail
            print("[EyeTracker] No image specified. Generating sample YouTube thumbnail for demonstration...")
            image_path = generate_youtube_thumbnail("sample_demo_creative.png")
        else:
            parser.error("Please specify an input image with --image <path> or use --demo.")

    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    tracker = EyeTracker()
    pil_img = Image.open(image_path).convert("RGB")

    # Run analysis
    result = tracker.analyze(pil_img, num_fixation_points=args.points)
    result.image_path = image_path

    # Save artifacts
    base_name = Path(image_path).stem
    saved = tracker.render_all_outputs(
        orig_img=pil_img,
        result=result,
        output_dir=args.output_dir,
        colormap=args.colormap,
        alpha=args.alpha,
        base_name=base_name,
    )

    if not args.quiet:
        print_audit_report(result)
        print("\n[Generated Artifacts]")
        for k, v in saved.items():
            print(f"  • {k:<20}: {v}")

    if args.json:
        report_data = {
            "image_path": result.image_path,
            "dimensions": {"width": result.image_width, "height": result.image_height, "aspect_ratio": result.aspect_ratio},
            "scores": {
                "clutter_score": result.clutter_score,
                "focal_dominance_score": result.focal_dominance_score,
                "rule_of_thirds_score": result.rule_of_thirds_score,
                "color_contrast_score": result.color_contrast_score,
            },
            "fixation_scanpath": [asdict(fp) for fp in result.fixation_points],
            "social_platforms": [asdict(pa) for pa in result.platform_audits],
            "saved_files": saved,
        }
        print("\n" + json.dumps(report_data, indent=2, default=_json_serializer))


if __name__ == "__main__":
    main()
