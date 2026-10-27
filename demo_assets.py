"""
demo_assets.py - Synthetic creative assets generator for EyeTracker testing.
Generates realistic social media images, YouTube thumbnails, and marketing creatives.
"""

import math
from PIL import Image, ImageDraw, ImageFont


def get_font(size: int, bold: bool = False):
    """Attempt to load a system TrueType font, falling back to default."""
    font_paths = [
        "/usr/share/fonts/Adwaita/AdwaitaSans-Bold.ttf" if bold else "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    ]
    for p in font_paths:
        try:
            return ImageFont.truetype(p, size)
        except (OSError, IOError):
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def generate_youtube_thumbnail(path: str = "demo_youtube_thumb.png") -> str:
    """Generate a realistic 1280x720 16:9 YouTube thumbnail with a face, headline, and badge."""
    w, h = 1280, 720
    img = Image.new("RGB", (w, h), (15, 18, 30))
    draw = ImageDraw.Draw(img)

    # 1. Vibrant diagonal gradient background
    for y in range(h):
        ratio = y / h
        r = int(18 + 55 * ratio)
        g = int(22 + 25 * (1.0 - ratio))
        b = int(50 + 95 * ratio)
        draw.line([(0, y), (w, y)], fill=(r, g, b))

    # Ambient glow on the left (behind face)
    for rad in range(260, 20, -20):
        alpha_val = int(35 * (1.0 - rad / 260.0))
        glow_col = (100 + alpha_val * 3, 50 + alpha_val * 2, 180 + alpha_val)
        draw.ellipse((100 - rad, 160 - rad, 600 + rad, 660 + rad), outline=glow_col, width=4)

    # 2. Stylized human face / avatar on left (x: 240..500, y: 180..490)
    # Hair silhouette
    draw.ellipse((220, 150, 520, 480), fill=(35, 20, 25))
    # Face skin oval (warm skin tone attracts strong visual attention)
    draw.ellipse((260, 210, 490, 480), fill=(245, 195, 160), outline=(225, 165, 130), width=4)
    # Forehead hair fringe
    draw.chord((240, 170, 510, 310), start=180, end=360, fill=(40, 25, 30))

    # Expressive high-contrast eyes
    # Left eye
    draw.ellipse((305, 300, 355, 335), fill=(255, 255, 255), outline=(60, 40, 35), width=2)
    draw.ellipse((322, 308, 345, 331), fill=(35, 95, 205))
    draw.ellipse((330, 314, 338, 322), fill=(255, 255, 255)) # catchlight
    # Right eye
    draw.ellipse((400, 300, 450, 335), fill=(255, 255, 255), outline=(60, 40, 35), width=2)
    draw.ellipse((417, 308, 440, 331), fill=(35, 95, 205))
    draw.ellipse((425, 314, 433, 322), fill=(255, 255, 255)) # catchlight

    # Eyebrows
    draw.arc((300, 280, 360, 310), start=200, end=340, fill=(45, 30, 25), width=4)
    draw.arc((395, 280, 455, 310), start=200, end=340, fill=(45, 30, 25), width=4)

    # Smile / open mouth
    draw.arc((340, 395, 420, 445), start=15, end=165, fill=(190, 40, 50), width=6)

    # 3. Bold High-Contrast Text on Right
    font_hero = get_font(62, bold=True)
    font_sub = get_font(34, bold=False)
    font_badge = get_font(26, bold=True)

    # Text shadow for readability
    draw.text((583, 193), "10X YOUR VIEWS", fill=(10, 10, 20), font=font_hero)
    draw.text((580, 190), "10X YOUR VIEWS", fill=(255, 235, 30), font=font_hero)

    draw.text((582, 282), "Eye-Tracking AI Secret", fill=(10, 10, 20), font=font_sub)
    draw.text((580, 280), "Eye-Tracking AI Secret", fill=(240, 245, 255), font=font_sub)

    # 4. Bright Action Badge / Button
    badge_x0, badge_y0 = 580, 360
    badge_x1, badge_y1 = 820, 430
    draw.rounded_rectangle((badge_x0, badge_y0, badge_x1, badge_y1), radius=16, fill=(235, 45, 55), outline=(255, 255, 255), width=3)
    draw.text((badge_x0 + 35, badge_y0 + 18), "WATCH NOW", fill=(255, 255, 255), font=font_badge)

    # 5. Accent graphic element (neon arrow pointing from text to badge)
    arrow_pts = [(900, 395), (960, 395), (940, 375), (960, 395), (940, 415)]
    draw.line([(880, 395), (950, 395)], fill=(0, 240, 220), width=5)
    draw.polygon([(940, 380), (970, 395), (940, 410)], fill=(0, 240, 220))

    img.save(path, quality=95)
    return path


def generate_bad_placement_thumbnail(path: str = "demo_bad_placement.png") -> str:
    """
    Generate a thumbnail where the critical call-to-action is in the bottom-right corner,
    which directly collides with YouTube's duration timestamp!
    """
    w, h = 1280, 720
    img = Image.new("RGB", (w, h), (25, 28, 40))
    draw = ImageDraw.Draw(img)

    font_hero = get_font(52, bold=True)
    font_sub = get_font(28, bold=False)
    font_badge = get_font(26, bold=True)

    draw.text((80, 80), "MISTAKE DEMO", fill=(255, 255, 255), font=font_hero)
    draw.text((80, 150), "Important button in dangerous bottom-right corner", fill=(180, 190, 210), font=font_sub)

    # Critical high-attention element placed in bottom-right corner (COLLISION ZONE)
    cx0, cy0 = 1040, 580
    cx1, cy1 = 1240, 680
    draw.rounded_rectangle((cx0, cy0, cx1, cy1), radius=12, fill=(255, 30, 60), outline=(255, 255, 0), width=4)
    draw.text((cx0 + 20, cy0 + 30), "CLICK HERE!", fill=(255, 255, 255), font=font_badge)

    img.save(path, quality=95)
    return path


if __name__ == "__main__":
    p1 = generate_youtube_thumbnail()
    print(f"Created: {p1}")
    p2 = generate_bad_placement_thumbnail()
    print(f"Created: {p2}")

