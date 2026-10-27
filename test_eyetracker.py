#!/usr/bin/env python3
"""
Unit and integration tests for EyeTracker.
Tests:
  - Saliency prediction shapes, normalization, and bounds
  - Colormap generators (Turbo, Jet, Inferno, Hot)
  - Scanpath extraction and Gaussian inhibition of return
  - Social media platform audit metrics, safe zones, and collision detection
  - Image rendering, overlays, and artifact generation
"""

import json
import os
import shutil
import tempfile
import unittest
import numpy as np
from PIL import Image

from eyetracker import (
    Colormaps,
    EyeTracker,
    HeatmapRenderer,
    PLATFORM_SPECS,
    SaliencyPredictor,
    ScanpathExtractor,
    SocialMediaAuditor,
)


class TestColormaps(unittest.TestCase):
    def test_colormap_shapes_and_ranges(self):
        x = np.linspace(0.0, 1.0, 100).reshape((10, 10))
        for name in ["turbo", "jet", "inferno", "hot"]:
            rgb = Colormaps.apply(x, name)
            self.assertEqual(rgb.shape, (10, 10, 3))
            self.assertEqual(rgb.dtype, np.uint8)
            self.assertTrue(np.all(rgb >= 0))
            self.assertTrue(np.all(rgb <= 255))


class TestSaliencyPredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = SaliencyPredictor(canonical_size=200)

    def test_saliency_dimensions_and_bounds(self):
        w, h = 320, 240
        dummy_img = Image.new("RGB", (w, h), (100, 120, 150))
        # Add a bright high-contrast circle
        from PIL import ImageDraw
        draw = ImageDraw.Draw(dummy_img)
        draw.ellipse((80, 60, 160, 140), fill=(255, 255, 0))

        sal = self.predictor.predict(dummy_img)
        self.assertEqual(sal.shape, (h, w))
        self.assertTrue(np.all(sal >= 0.0))
        self.assertTrue(np.all(sal <= 1.0))
        self.assertAlmostEqual(float(sal.max()), 1.0, places=3)

    def test_peak_detected_at_contrast_region(self):
        w, h = 400, 300
        # Dark canvas with one bright yellow box in top-left
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[40:100, 50:120] = [255, 240, 20]
        img = Image.fromarray(arr)

        sal = self.predictor.predict(img)
        max_idx = np.unravel_index(np.argmax(sal), sal.shape)
        py, px = max_idx

        # The peak should be inside or immediately around the bright box
        self.assertTrue(30 <= py <= 110, f"Expected peak y in [30, 110], got {py}")
        self.assertTrue(40 <= px <= 130, f"Expected peak x in [40, 130], got {px}")


class TestScanpathExtractor(unittest.TestCase):
    def test_scanpath_point_count_and_spacing(self):
        h, w = 600, 800
        sal = np.zeros((h, w), dtype=np.float32)
        y, x = np.ogrid[:h, :w]

        # 3 distinct gaussian peaks
        sal += 1.0 * np.exp(-((x - 200)**2 + (y - 200)**2) / (2 * 40**2))
        sal += 0.8 * np.exp(-((x - 600)**2 + (y - 200)**2) / (2 * 40**2))
        sal += 0.6 * np.exp(-((x - 400)**2 + (y - 450)**2) / (2 * 40**2))

        points = ScanpathExtractor.extract(sal, num_points=3)
        self.assertEqual(len(points), 3)

        # Check order
        self.assertEqual(points[0].order, 1)
        self.assertEqual(points[1].order, 2)
        self.assertEqual(points[2].order, 3)

        # Fixation #1 should be near (200, 200)
        self.assertAlmostEqual(points[0].x, 200, delta=15)
        self.assertAlmostEqual(points[0].y, 200, delta=15)

        # Fixation #2 should be near (600, 200)
        self.assertAlmostEqual(points[1].x, 600, delta=15)
        self.assertAlmostEqual(points[1].y, 200, delta=15)

        # Check that coordinates are native ints/floats for JSON safety
        self.assertIsInstance(points[0].x, int)
        self.assertIsInstance(points[0].norm_x, float)


class TestSocialMediaAuditor(unittest.TestCase):
    def test_platform_specs_loaded(self):
        self.assertGreaterEqual(len(PLATFORM_SPECS), 6)
        names = [p.platform_name for p in PLATFORM_SPECS]
        self.assertIn("YouTube", names)
        self.assertIn("Instagram", names)
        self.assertIn("TikTok", names)

    def test_youtube_timestamp_occlusion_detection(self):
        w, h = 1280, 720
        sal = np.zeros((h, w), dtype=np.float32)
        y, x = np.ogrid[:h, :w]

        # Concentrated gaze in bottom-right corner (YouTube duration zone)
        sal += np.exp(-((x - 1150)**2 + (y - 650)**2) / (2 * 50**2))
        sal /= sal.sum()

        fixations = ScanpathExtractor.extract(sal, num_points=2)
        audits, clutter, focal, rot, contrast = SocialMediaAuditor.audit_image(sal, w, h, fixations)

        yt_audit = next(a for a in audits if a.platform_name == "YouTube")
        # Should detect significant occlusion by Duration Timestamp Badge
        self.assertTrue(len(yt_audit.danger_zone_collisions) > 0)
        self.assertIn("Duration Timestamp Badge", yt_audit.danger_zone_collisions[0])
        self.assertLess(yt_audit.safe_zone_pct, 50.0)


class TestEyeTrackerFullPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_analyze_and_render_all(self):
        img = Image.new("RGB", (640, 360), (30, 40, 50))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle((50, 50, 200, 200), fill=(250, 200, 50))
        draw.text((60, 80), "TEST HOOK", fill=(0, 0, 0))

        tracker = EyeTracker(canonical_size=200)
        res = tracker.analyze(img, num_fixation_points=4)

        self.assertEqual(res.image_width, 640)
        self.assertEqual(res.image_height, 360)
        self.assertGreaterEqual(len(res.fixation_points), 1)
        self.assertGreaterEqual(len(res.platform_audits), 6)

        # Test rendering all outputs
        saved = tracker.render_all_outputs(
            orig_img=img,
            result=res,
            output_dir=self.temp_dir,
            colormap="turbo",
            alpha=0.6,
            base_name="unit_test",
        )

        self.assertIn("heatmap_overlay", saved)
        self.assertIn("scanpath", saved)
        self.assertIn("saliency_raw", saved)
        self.assertIn("social_audit_grid", saved)
        self.assertIn("json_report", saved)

        for name, filepath in saved.items():
            self.assertTrue(os.path.exists(filepath), f"Missing generated file: {filepath}")
            self.assertGreater(os.path.getsize(filepath), 100)

        # Validate JSON content
        with open(saved["json_report"]) as f:
            data = json.load(f)
            self.assertIn("dimensions", data)
            self.assertIn("social_platforms", data)
            self.assertEqual(data["dimensions"]["width"], 640)


if __name__ == "__main__":
    unittest.main()

