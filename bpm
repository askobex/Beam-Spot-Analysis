# ========================================================================
# Beam Diagnostics SDK — Full OOP Framework (Component-Based)
# BioXAS Beamline — Canadian Light Source
#
# Author: Peter Ufondu
# Version: 2026.1
#
# Features:
# Camera → ROIManager → BeamAnalyzer → BeamTracker
#                                       ├── LiveDashboardQt
#                                       ├── BeamDiagnostics (Batch)
#                                       ├── LiveRecorder
#                                       └── BPM (Tk viewer)
# Camera → ROI → BeamAnalyzer → BeamTracker
                                 # ├── LiveDashboardQt
                                 # │     ├── LiveRollingFFT (Welch PSD)
                                 # │     └── Operator diagnostics
                                 # ├── BeamDiagnostics (batch)
                                 # └── BPM (reference viewer)
# Core OOP Modules:
#   CameraInterface     — Abstract base
#   FastCamera          — USB/GStreamer threaded reader(optional)
#   ThorCamera          — Thorlabs SDK camera 
#   ROIManager          — Smart center + adaptive ROI
#   BeamAnalyzer        — All measurement methods
#   BeamTracker         — Core processing engine
#   BeamDiagnostics     — Batch processor
#   LiveDashboard       — Real‑time 4-panel dashboard
#   Utils               — Math helpers
# ======================================================================
# Standard Library
# ======================================================================
import os
import sys
import time
import threading
import asyncio
import queue
import math
import concurrent.futures
from typing import Optional, Tuple
from contextlib import suppress

# ======================================================================
# Core Scientific / Data Stack
# ======================================================================
import numpy as np
import cv2
import h5py
import pandas as pd

# ======================================================================
# Plotting & Visualization (Matplotlib / PyQtGraph)
# ======================================================================
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

import pyqtgraph as pg

# ======================================================================
# Jupyter / Notebook Utilities (Optional)
# ======================================================================
try:
    import ipywidgets as widgets
    from IPython.display import display
except Exception:
    widgets = None
    display = None

# Optional Bokeh (Notebook embedding)
try:
    from bokeh.io import output_notebook
    output_notebook()
except Exception:
    pass

# ======================================================================
# Qt (Live Dashboard)
# ======================================================================
from PySide6.QtCore import QTimer, QRectF, Qt
from PySide6.QtWidgets import (
    QWidget,
    QPushButton,
    QGridLayout,
    QApplication,QLabel, QVBoxLayout
)
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from scipy.signal import find_peaks, welch
from scipy.ndimage import gaussian_filter1d
# ======================================================================
# Tkinter (BPM / Live Camera Viewer)
# ======================================================================
import tkinter as tk
from PIL import Image, ImageTk

# ======================================================================
# Thorlabs Scientific Camera SDK
# ======================================================================
from thorlabs_tsi_sdk.tl_camera import (
    TLCameraSDK,
    TLCamera,
    Frame,
    OPERATION_MODE,
)
from thorlabs_tsi_sdk.tl_camera_enums import SENSOR_TYPE
from thorlabs_tsi_sdk.tl_color_enums import FORMAT
from thorlabs_tsi_sdk.tl_mono_to_color_processor import (
    MonoToColorProcessorSDK,
)
from collections import deque
# ======================================================================
# Optional SciPy (Gaussian fitting)
# ======================================================================
try:
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False
# ======================================================================
# Rotation
# ======================================================================
def rotate_image(image, theta_deg):
    h, w = image.shape
    M = cv2.getRotationMatrix2D(
        center=(w / 2, h / 2),
        angle=theta_deg,
        scale=1.0
    )
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR
    )

def rotated_profiles_fast(image, theta):
    h, w = image.shape
    yy, xx = np.indices((h, w))

    xx = xx - w / 2
    yy = yy - h / 2

    u =  np.cos(theta) * xx + np.sin(theta) * yy
    v = -np.sin(theta) * xx + np.cos(theta) * yy

    # Discretize coordinates
    u_idx = np.round(u - u.min()).astype(np.int32)
    v_idx = np.round(v - v.min()).astype(np.int32)

    nu = u_idx.max() + 1
    nv = v_idx.max() + 1

    prof_u = np.bincount(
        u_idx.ravel(),
        weights=image.ravel(),
        minlength=nu
    )

    prof_v = np.bincount(
        v_idx.ravel(),
        weights=image.ravel(),
        minlength=nv
    )

    return prof_u, prof_v
# ========================================================================
# Backend helpers
# ========================================================================
def detect_backend() -> str:
    try:
        b = matplotlib.get_backend().lower()
    except Exception:
        return "other"
    if "ipympl" in b or "widget" in b:
        return "ipympl"
    if "inline" in b or "nbagg" in b:
        return "inline"
    return "other"


def enable_widget_backend():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            ip.run_line_magic("matplotlib", "widget")
    except Exception:
        pass
# ========================================================================
# Utility Functions
# ========================================================================

class Utils:
    @staticmethod
    def sigma_to_fwhm(sigma: float) -> float:
        return 2 * np.sqrt(2 * np.log(2)) * float(sigma)

    @staticmethod
    def ema(prev: Optional[float], new: float, a: float = 0.3) -> float:
        if prev is None:
            return float(new)
        return (1 - a) * float(prev) + a * float(new)

    @staticmethod
    def downsample(img: np.ndarray, factor: int) -> np.ndarray:
        if factor is None or factor <= 1:
            return img
        h, w = img.shape[:2]
        return cv2.resize(img, (w // factor, h // factor), interpolation=cv2.INTER_AREA)

    @staticmethod
    def snr(roi: np.ndarray) -> float:
        p50 = np.percentile(roi, 50)
        p95 = np.percentile(roi, 95)
        mad = np.median(np.abs(roi - p50)) + 1e-9
        return (p95 - p50) / mad


# ========================================================================
# Camera Interfaces
# ========================================================================

class CameraInterface:
    """Abstract camera interface class."""
    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError


# ========================================================================
#   FastCamera          — USB/GStreamer threaded reader(optional)
# ========================================================================
class FastCamera(CameraInterface):
    """
    Threaded high-speed USB / GStreamer / V4L2 camera.
    Always returns grayscale frames.
    """

    def __init__(self, source="/dev/video2", use_gst=False, width=640, height=480, fps=100):
        # source can be "/dev/video2", "/dev/video4", or an int (0,1,2,...)
        if use_gst:
            gst = (
                f"v4l2src device={source} ! "
                f"video/x-raw, width={width}, height={height}, framerate={fps}/1 ! "
                "videoconvert ! appsink max-buffers=1 drop=true"
            )
            self.cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)

        else:
            # If source is a path, convert it to an index
            if isinstance(source, str) and source.startswith("/dev/video"):
                # Extract the number after /dev/videoX
                source_idx = int(source.replace("/dev/video", ""))
            else:
                source_idx = source

            self.cap = cv2.VideoCapture(source_idx, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)

        if not self.cap.isOpened():
            raise RuntimeError(f"FastCamera: cannot open camera at {source}")

        self.running = True
        self.lock = threading.Lock()
        self.frame = None

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frm = self.cap.read()
            if not ok:
                continue
            if len(frm.shape) == 3:
                frm = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
            with self.lock:
                self.frame = frm

    def read(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

# ========================================================================
#   ThorCamera          — Thorlabs SDK camera 
# ========================================================================
class ThorCamera(CameraInterface):
    """Thorlabs Scientific Camera (SciCam SDK)."""

    def __init__(self, index=0, exposure_us=10000):
        self.sdk = TLCameraSDK()

        # detect cameras
        serials = self.sdk.discover_available_cameras()
        if not serials:
            raise RuntimeError("No Thorlabs cameras found.")

        # open camera
        self.camera = self.sdk.open_camera(serials[index])

        # store metadata
        self.is_color = (self.camera.camera_sensor_type == SENSOR_TYPE.BAYER)
        self.width = self.camera.image_width_pixels
        self.height = self.camera.image_height_pixels
        self.bit_depth = self.camera.bit_depth

        # configure basic settings
        self.camera.exposure_time_us = exposure_us
        # self.camera.frames_per_trigger_zero_for_unlimited = 0
        self.camera.image_poll_timeout_ms = 1000
        # self.camera.operation_mode = OPERATION_MODE.CONTINUOUS
        self.camera.operation_mode = OPERATION_MODE.SOFTWARE_TRIGGERED
        # continuous mode is enabled simply by:
        # self.camera.frame_rate_control_value = 10
        # self.camera.is_frame_rate_control_enabled = True
        self.camera.model
        self.camera.serial_number          # sometimes available depending on SDK version
        self.camera.sensor_width_pixels
        self.camera.sensor_height_pixels
        self.camera.exposure_time_us
        self.camera.frame_time_us
        self.camera.operation_mode
        self.camera.bit_depth

        # color processor (if needed)
        self.color_sdk = None
        self.color_proc = None
        if self.is_color:
            self.color_sdk = MonoToColorProcessorSDK()
            self.color_proc = self.color_sdk.create_mono_to_color_processor(
                self.camera.camera_sensor_type,
                self.camera.color_filter_array_phase,
                self.camera.get_color_correction_matrix(),
                self.camera.get_default_white_balance_matrix(),
                self.camera.bit_depth
            )
            self.color_proc.output_format = FORMAT.BGR_PIXEL  # OpenCV format

        # arm camera
        self.camera.arm(2)
        self.camera.issue_software_trigger()

    def close(self):
    
        # 1. Close camera handle FIRST
        try:
            if hasattr(self, "cam") and self.cam is not None:

                self.camera.disarm()
                self.camera.dispose()
        except Exception as e:
            print("Camera close error:", e)
    
        # 2. Then close SDK SECOND
        try:
            if hasattr(self, "sdk") and self.sdk is not None:
                self.sdk.dispose()   # or shutdown()
        except Exception as e:
            print("SDK close error:", e)

    def read(self) -> Optional[np.ndarray]:
        """Return latest frame as np.ndarray or None."""

        frame = self.camera.get_pending_frame_or_null()
        if frame is None:
            return None
        return frame.image_buffer.copy()

    def stop(self):
        """Clean shutdown."""
        try:
            self.camera.disarm()
        except Exception:
            pass

        try:
            self.camera.dispose()
        except Exception:
            pass

        if self.is_color and self.color_proc is not None:
            try:
                self.color_proc.dispose()
            except Exception:
                pass
            try:
                self.color_sdk.dispose()
            except Exception:
                pass

        self.sdk.dispose()
# ========================================================================
# ROI Manager
# ========================================================================

class ROIManager:
    """
    Smart + Adaptive ROI manager:
        • Smart init (CLAHE + threshold)
        • Adaptive sizing based on sigma hints
        • Edge detection → forces re-center
    """

    def __init__(self, default_half=100, min_half=20, margin=12, k=4.0):
        self.default_half = default_half
        self.min_half = min_half
        self.margin = margin
        self.k = k

    def smart_center(self, gray: np.ndarray) -> Tuple[int, int]:
        img = gray.astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        thr = np.percentile(blur, 97)
        mask = blur > thr

        if np.any(mask):
            num, labels, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
            if num > 1:
                best = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                cx, cy = cent[best]
                return int(cy), int(cx)

        iy, ix = np.unravel_index(int(np.argmax(blur)), blur.shape)
        return int(iy), int(ix)

    def get_roi(self, gray, cx, cy, sigma_hint=None):
        h, w = gray.shape
        if sigma_hint is None:
            half = self.default_half
        else:
            sx, sy = sigma_hint
            half = int(np.clip(self.k * max(sx, sy) + self.margin,
                               self.min_half, min(h, w) // 2))

        cy = int(np.clip(cy, 0, h - 1))
        cx = int(np.clip(cx, 0, w - 1))
        y0 = max(0, cy - half); y1 = min(h, cy + half)
        x0 = max(0, cx - half); x1 = min(w, cx + half)

        return gray[y0:y1, x0:x1], (x0, y0, x1, y1)

    def needs_redetect(self, xo, yo, w, h, snr, fit_ok):
        margin = 0.08
        edge = (
            xo < margin * w or xo > (1 - margin) * w or
            yo < margin * h or yo > (1 - margin) * h
        )
        return edge or snr < 3.0 or not fit_ok

# ========================================================================
# LiveRollingFFT — Welch-style averaged FFT for LiveDashboard
# ========================================================================
class LiveRollingFFT:
    """
    Rolling FFT with Welch-style PSD averaging.
    Lightweight enough for LiveDashboardQt.
    """

    def __init__(self, fs, window_size, avg_blocks=8):
        self.fs = fs
        self.window_size = int(window_size)
        self.avg_blocks = int(avg_blocks)

        # ✅ buffer length MUST be integer number of samples
        self.buffer = deque(maxlen=self.window_size * self.avg_blocks)

        self.psd_accum = None
        self.count = 0

    def push(self, x):
        self.buffer.append(float(x))

    def compute(self):
        if len(self.buffer) < self.window_size:
            return None, None

        # take most recent window
        data = np.array(self.buffer)[-self.window_size:]

        f, pxx = welch(
            data,
            fs=self.fs,
            nperseg=self.window_size,
            detrend="constant",
            window="hann",
        )

        if self.psd_accum is None:
            self.psd_accum = np.zeros_like(pxx)

        self.psd_accum += pxx
        self.count += 1

        return f, self.psd_accum / self.count

    def reset(self):
        """Clear accumulated PSD so the next FFT window is independent."""
        self.psd_accum = None
        self.count = 0

# ========================================================================
# Beam Analyzer (moments, 1D, 2D Gaussian fits)
# ========================================================================

class BeamAnalyzer:
    def __init__(self, method="moments"):
        self.method = method
# ========================================================================
# MOMENTS 
# ========================================================================
    def _moments(self, roi):
        R = np.clip(roi, 0, None)
        s = R.sum()
        if s <= 0:
            h, w = R.shape
            return (w - 1) / 2, (h - 1) / 2, w / 10, h / 10

        px = R.sum(axis=0)
        py = R.sum(axis=1)
        xs = np.arange(px.size)
        ys = np.arange(py.size)

        x0 = (px @ xs) / s
        y0 = (py @ ys) / s

        sx = np.sqrt(max((px @ (xs - x0) ** 2) / s, 1e-9))
        sy = np.sqrt(max((py @ (ys - y0) ** 2) / s, 1e-9))
        return float(x0), float(y0), float(sx), float(sy)

# ========================================================================
# 1D FIT 
# ========================================================================
    def _gauss1d(self, x, amp, x0, sigma, off):
        return off + amp * np.exp(-0.5 * ((x - x0) / sigma) ** 2)

    def _projection_fit(self, roi):
        if not _HAS_SCIPY:
            return self._moments(roi)

        R = np.clip(roi, 0, None)
        px = R.sum(axis=0)
        py = R.sum(axis=1)
        xs = np.arange(px.size)
        ys = np.arange(py.size)

        def est(p, coords):
            s = p.sum()
            if s <= 0:
                c = coords[len(coords)//2]
                return c, len(coords)/10
            m = (p @ coords) / s
            v = (p @ (coords - m)**2) / s
            return m, np.sqrt(max(v, 1e-9))

        x0i, sxi = est(px, xs)
        y0i, syi = est(py, ys)

        amp_x = max(px.max() - np.median(px), 1.0)
        off_x = np.median(px)
        p0x = [amp_x, x0i, max(sxi, 0.5), off_x]

        amp_y = max(py.max() - np.median(py), 1.0)
        off_y = np.median(py)
        p0y = [amp_y, y0i, max(syi, 0.5), off_y]

        try:
            fx, _ = curve_fit(self._gauss1d, xs, px, p0=p0x, maxfev=2000)
        except Exception:
            fx = p0x
        try:
            fy, _ = curve_fit(self._gauss1d, ys, py, p0=p0y, maxfev=2000)
        except Exception:
            fy = p0y

        return fx[1], fy[1], abs(fx[2]), abs(fy[2])
        
# ========================================================================
# 2D FIT 
# ========================================================================
    def _gaussian_2d(self, coords, A, x0, y0, sx, sy, B):
        x, y = coords
        return (A*np.exp(-(((x-x0)**2)/(2*sx**2) + ((y-y0)**2)/(2*sy**2))) + B).ravel()

    def _moments_guess(self, I):
        I = np.nan_to_num(I, nan=0.0)
        s = I.sum()
        if s <= 0:
            h, w = I.shape
            return 1.0, (w-1)/2, (h-1)/2, w/10, h/10, 0.0

        px = I.sum(axis=0)
        py = I.sum(axis=1)
        xs = np.arange(px.size)
        ys = np.arange(py.size)

        x0 = (px @ xs) / s
        y0 = (py @ ys) / s
        sx = np.sqrt(max((px @ (xs - x0)**2) / s, 1e-9))
        sy = np.sqrt(max((py @ (ys - y0)**2) / s, 1e-9))
        B0 = np.percentile(I, 10)
        A0 = max(I.max() - B0, 1.0)
        return A0, x0, y0, sx, sy, B0

    def _fit2d(self, roi):
        if not _HAS_SCIPY:
            return self._moments(roi)

        I = np.clip(roi, 0, None)
        h, w = I.shape
        A0, x0, y0, sx0, sy0, B0 = self._moments_guess(I)

        yy, xx = np.indices((h, w))
        coords = (xx.astype(float), yy.astype(float))

        try:
            p, _ = curve_fit(
                self._gaussian_2d, coords, I.ravel(),
                p0=[A0, x0, y0, sx0, sy0, B0],
                maxfev=8000
            )
            return float(p[1]), float(p[2]), abs(float(p[3])), abs(float(p[4]))
        except Exception:
            return x0, y0, abs(sx0), abs(sy0)
# ========================================================================
#  PUBLIC API 
# ========================================================================
    def analyze(self, roi: np.ndarray):
        if self.method == "moments":
            return self._moments(roi)
        elif self.method == "projection_fit":
            return self._projection_fit(roi)
        elif self.method == "gaussian_2d":
            return self._fit2d(roi)
        else:
            raise ValueError("Unknown method")
# ========================================================================
# BeamTracker (core engine for live + batch)
# ========================================================================

class BeamTracker:
    """Core tracking logic shared by Live and Batch modes."""

    def __init__(self, camera: CameraInterface,
                 analyzer: BeamAnalyzer,
                 roi_mgr: ROIManager,
                 pixel_size_um=4.8,
                 screen_distance_m=0.13,
                 downsample=1,
                 max_miss=10):

        self.camera = camera
        self.analyzer = analyzer
        self.roi_mgr = roi_mgr
        self.pixel = pixel_size_um
        self.screen = screen_distance_m
        self.down = downsample
        self.max_miss = max_miss

    def process_frame(self, gray, cx, cy, sx_hint, sy_hint, miss_count):
        # ROI
        roi, (x0, y0, x1, y1) = self.roi_mgr.get_roi(
            gray, cx, cy,
            sigma_hint=(sx_hint, sy_hint) if sx_hint is not None else None
        )
        roi = roi.astype(np.float64)
        roi = roi - np.median(roi)
        roi_sub = np.clip(roi, 0, None)
        
        snr = Utils.snr(roi)
        roi_proc = Utils.downsample(roi, self.down)
        scale = 1.0 if self.down <= 1 else self.down

        xo, yo, sx, sy = self.analyzer.analyze(roi_proc)
        xo *= scale; yo *= scale
        sx *= scale; sy *= scale

        beam_x = x0 + xo
        beam_y = y0 + yo

        fwhm_x_pix = Utils.sigma_to_fwhm(abs(sx))
        fwhm_y_pix = Utils.sigma_to_fwhm(abs(sy))
        fwhm_x_um = fwhm_x_pix * self.pixel
        fwhm_y_um = fwhm_y_pix * self.pixel

        div_x = fwhm_x_um / (self.screen * 1e6)
        div_y = fwhm_y_um / (self.screen * 1e6)

        intensity = float(np.clip(roi, 0, None).sum())

        fit_ok = all(np.isfinite([xo, yo, sx, sy]))
        h_roi, w_roi = roi.shape
        if self.roi_mgr.needs_redetect(xo, yo, w_roi, h_roi, snr, fit_ok):
            miss_count += 1
        else:
            miss_count = 0

        if miss_count >= self.max_miss:
            small = Utils.downsample(gray, 2)
            cy_s, cx_s = self.roi_mgr.smart_center(small)
            return beam_x, beam_y, fwhm_x_um, fwhm_y_um, div_x, div_y, intensity, miss_count, (cx_s*2, cy_s*2), (sx, sy), roi,roi_sub, (x0, y0, x1, y1)

        # return beam_x, beam_y, fwhm_x_um, fwhm_y_um, div_x, div_y, intensity, miss_count, None, (sx, sy)
        return (
            beam_x, beam_y,
            fwhm_x_um, fwhm_y_um,
            div_x, div_y,
            intensity,
            miss_count,
            None,
            (sx, sy),
            roi,              # ✅ add ROI
            roi_sub,
            (x0, y0, x1, y1), # ✅ add ROI box
        )


# ========================================================================
# Batch Processing: BeamDiagnostics
# ========================================================================

class BeamDiagnostics:
    def __init__(self, tracker: BeamTracker):
        self.tracker = tracker

    def analyze_video(self, path, save_prefix="beam_output", plot_every=0):
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Cannot read video.")

        gray0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cx, cy = self.tracker.roi_mgr.smart_center(gray0)

        sx_hint = sy_hint = None
        miss = 0
        data = []
        fig = None

        if plot_every > 0:
            fig, ax = plt.subplots()

        frame_idx = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            (bx, by, fx, fy, dx, dy, inten,
             miss, recenter, (sx_hint, sy_hint)) = self.tracker.process_frame(
                gray, cx, cy, sx_hint, sy_hint, miss
            )

            if recenter:
                cx, cy = recenter
            else:
                cx = Utils.ema(cx, bx, a=0.35)
                cy = Utils.ema(cy, by, a=0.35)

            data.append([frame_idx, bx, by, fx, fy, dx, dy, inten])

            if fig and frame_idx % plot_every == 0:
                ax.clear()
                ax.scatter(bx, by, c='cyan')
                ax.set_title(f"Frame {frame_idx}")
                plt.pause(0.001)

            frame_idx += 1

        cap.release()

        df = pd.DataFrame(data, columns=[
            "frame", "beam_x", "beam_y",
            "FWHM_x_um", "FWHM_y_um",
            "div_x", "div_y", "intensity"
        ])

        df.to_csv(save_prefix + ".csv", index=False)
        with h5py.File(save_prefix + ".h5", "w") as f:
            ds = f.create_dataset("beam", data=df.values)
            ds.attrs["columns"] = df.columns.astype(str).tolist()

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1); plt.plot(df["FWHM_x_um"]); plt.plot(df["FWHM_y_um"]); plt.title("Size (µm)")
        plt.subplot(1, 3, 2); plt.plot(df["intensity"]); plt.title("Intensity")
        plt.subplot(1, 3, 3); plt.plot(df["beam_x"]); plt.plot(df["beam_y"]); plt.title("Drift")
        plt.tight_layout()
        plt.savefig(save_prefix + "_summary.png", dpi=150)
        plt.close()

        return df

# ========================================================================
# Non-blocking camera read inside background thread
# ========================================================================
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
async def cam_read_async(cam):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, cam.read)

# ========================================================================
# LiveRecorder — ROI-only MP4 writer 
# ========================================================================
class LiveRecorder:
    """
    Fixed-size ROI MP4 recorder.
    """
    def __init__(self, path, fps):
        self.path = path
        self.fps = fps
        self.writer = None
        self.size = None

    def start(self, roi_shape):
        h, w = roi_shape
        self.size = (w, h)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            self.path, fourcc, self.fps, self.size, isColor=True
        )
        if not self.writer.isOpened():
            raise RuntimeError("VideoWriter failed")

    def write(self, roi_u8):
        if self.writer is None:
            return

        if roi_u8.shape[:2] != (self.size[1], self.size[0]):
            roi_u8 = cv2.resize(roi_u8, self.size)

        frame = cv2.cvtColor(roi_u8, cv2.COLOR_GRAY2BGR)
        self.writer.write(frame)

    def stop(self):
        if self.writer:
            self.writer.release()
            self.writer = None
            
# ========================================================================
# LiveDashboardQt — Beam Diagnostics (4 panels + ROI video + CSV sync)
# ========================================================================
class LiveDashboardQt(QWidget):
    """
    Full Live Beam Diagnostics Dashboard
    """

    # ==================================================
    def __init__(self, tracker, csv_path=None, record_path=None, window_size=None):
        super().__init__()

        # ================= Camera =================
        self.tracker = tracker
        self.cam = tracker.camera
        self.sdk_cam = tracker.camera.camera

        self.frame_time_us = self.sdk_cam.frame_time_us
        self.dt = self.frame_time_us * 1e-6
        self.fs = 1e6 / self.frame_time_us

        # ================= State ==================
        self.cx = self.cy = None
        self.sx_hint = self.sy_hint = None
        self.miss = 0
        self.frame_idx = 0
        self.latest_frame = None
        self.stop_requested = False
        self.window_size =  window_size
        
        # ================= FFT engines =================
        self.fft_fx = LiveRollingFFT(self.fs, self.window_size, avg_blocks=8)
        self.fft_fy = LiveRollingFFT(self.fs, self.window_size, avg_blocks=8)
        self.fft_bx = LiveRollingFFT(self.fs, self.window_size, avg_blocks=8)
        self.fft_by = LiveRollingFFT(self.fs, self.window_size, avg_blocks=8)

        # ================= Buffers ================
        self.X, self.INT = [], []
        self.BX, self.BY = [], []
        self.DX, self.DY = [], []
        self.FX, self.FY = [], []
        
        self.csv_path = csv_path
        self.record_csv = csv_path is not None
        self.records = []
        self.live_records = []
        
        self.record_video = record_path is not None
        self.recorder = LiveRecorder(record_path, self.fs) if record_path else None
        self.rec_started = False

        # ================= UI =====================

        self.setWindowTitle("BioXAS Beam Diagnostics — Live View")
        self.resize(800, 800)
        grid = QGridLayout(self)
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setSpacing(6)


        self.btn_stop = QPushButton("⛔ STOP")
        self.btn_stop.clicked.connect(self.stop)

        self.lbl_acq = QLabel()
        self.lbl_acq.setStyleSheet("""
            QLabel { font-size:12px; background:#f0f0f0;
                     padding:4px 8px; border-radius:4px; }
        """)

        grid.addWidget(self.btn_stop, 0, 0)
        grid.addWidget(self.lbl_acq, 0, 1, 1, 2)


        # ================= ROI ====================
        self.img_view = pg.PlotWidget(title="ROI (heatmap)")
        self.img_view.setAspectLocked(True)
        self.img_item = pg.ImageItem(axisOrder="row-major")
        self.img_view.addItem(self.img_item)
        self.img_item.setLookupTable(
            pg.colormap.get("inferno").getLookupTable(alpha=True, mode="byte")
        )
        self.beam_dot = pg.ScatterPlotItem(size=10, brush="c", pen=None)
        self.img_view.addItem(self.beam_dot)
        grid.addWidget(self.img_view, 1, 1)

        # ================= Other Plots ============
        self.plot_int = pg.PlotWidget(title="Intensity")
        self.int_line = self.plot_int.plot(pen="m")

        self.plot_px = pg.PlotWidget(title="Beam Profile X (px)")
        self.plot_py = pg.PlotWidget(title="Beam Profile Y (px)")
        # --- Rotate Beam Profile Y logically (no widget rotation) ---
        pi = self.plot_py.getPlotItem()
        
        # Show top and right axes
        pi.showAxis('top')
        pi.showAxis('right')
        
        # Hide bottom & left
        pi.hideAxis('top')
        pi.hideAxis('right')
        pi.hideAxis('bottom')
        
        # Labels
        # pi.setLabel('bottom', 'Intensity (a.u.)')
        pi.setLabel('left', 'Y position (px)')
        
        # Invert Y so +up matches image coordinates
        pi.invertY(False)
        
        # Optional: lock aspect so it looks like a side projection
        pi.getViewBox().setAspectLocked(False)
        self.line_px = self.plot_px.plot(pen="y")
        self.line_py = self.plot_py.plot(pen="c")
        # self.gfit_px = self.plot_px.plot(pen=pg.mkPen("r", width=2, style=Qt.DashLine))
        # self.gfit_py = self.plot_py.plot(pen=pg.mkPen("r", width=2, style=Qt.DashLine))

        self.plot_size = pg.PlotWidget(title="Beam Size (µm)")
        self.fx_line = self.plot_size.plot(pen="y")
        self.fy_line = self.plot_size.plot(pen="c")

        self.plot_pos = pg.PlotWidget(title="Beam Drift (µm)")
        self.bx_line = self.plot_pos.plot(pen="y")
        self.by_line = self.plot_pos.plot(pen="c")
        
        # self.plot_div = pg.PlotWidget(title="Beam Divergence ")
        # self.dx_line = self.plot_div.plot(pen="y")
        # self.dy_line = self.plot_div.plot(pen="c")
        
        self.plot_fft_size = pg.PlotWidget(title="FFT — Beam Size")
        self.plot_fft_pos  = pg.PlotWidget(title="FFT — Drift")

        self.fft_fx_line = self.plot_fft_size.plot(pen="y")
        self.fft_fy_line = self.plot_fft_size.plot(pen="c")
        self.fft_fx_peaks = pg.ScatterPlotItem(size=8, brush="r")
        self.fft_fy_peaks = pg.ScatterPlotItem(size=8, brush="r")
        self.plot_fft_size.addItem(self.fft_fx_peaks)
        self.plot_fft_size.addItem(self.fft_fy_peaks)

        self.fft_bx_line = self.plot_fft_pos.plot(pen="y")
        self.fft_by_line = self.plot_fft_pos.plot(pen="c")
        self.fft_bx_peaks = pg.ScatterPlotItem(size=8, brush="r")
        self.fft_by_peaks = pg.ScatterPlotItem(size=8, brush="r")
        self.plot_fft_pos.addItem(self.fft_bx_peaks)
        self.plot_fft_pos.addItem(self.fft_by_peaks)


        grid.addWidget(self.plot_py, 1, 0)
        grid.addWidget(self.plot_int, 2, 0)
        grid.addWidget(self.plot_px, 2, 1)

        grid.addWidget(self.plot_pos, 3, 0)
        grid.addWidget(self.plot_size, 3, 1)
        # grid.addWidget(self.plot_div, 3, 1)
        grid.addWidget(self.plot_fft_pos, 4, 0)
        grid.addWidget(self.plot_fft_size, 4, 1)


        # ================= Mouse Hover ============
        self._mouse_proxies = []
        for plot in (
            self.img_view, self.plot_int, self.plot_px, self.plot_py,
            self.plot_size, self.plot_pos, self.plot_fft_size, self.plot_fft_pos
        ):
            plot.getViewBox().setAcceptHoverEvents(True)
            self._mouse_proxies.append(
                pg.SignalProxy(
                    plot.scene().sigMouseMoved,
                    rateLimit=60,
                    slot=lambda evt, p=plot: self._on_mouse_moved(evt, p)
                )
            )

        # ================= Threads ================
        threading.Thread(target=self._camera_loop, daemon=True).start()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_ui)
        self.timer.start(max(1, int(self.frame_time_us / 1000)))

    # ==================================================
    def _on_mouse_moved(self, evt, plot):
        """Mouse hover handler (correct per‑plot dispatch)."""
        mouse_point = plot.getViewBox().mapSceneToView(evt[0])
        title = plot.getPlotItem().titleLabel.text
        self.lbl_acq.setText(
            f'<span style="color:red;">'
            f"{title}: x={mouse_point.x():.3f}, y={mouse_point.y():.3f}"
            f'</span>'
        )

    # ==================================================
    def _camera_loop(self):
        while not self.stop_requested:
            frame = self.cam.read()
            if frame is not None:
                self.latest_frame = frame
            # time.sleep(0.0002)

    # ==================================================
    def _update_ui(self):
        if self.latest_frame is None:
            return
    
        gray = self.latest_frame
    
        # --------------------------------------------------
        # Initial center (first frame only)
        # --------------------------------------------------
        if self.cx is None:
            self.cx, self.cy = self.tracker.roi_mgr.smart_center(gray)
    
        # --------------------------------------------------
        # Single call: processing + ROI extraction
        # --------------------------------------------------
        (
            bx, by,                # beam position (pixels, full image)
            fx, fy,                # FWHM (µm)
            dx, dy,                  # divergence (unused here)
            inten,                 # intensity
            self.miss,             # miss counter
            recenter,              # None or (cx, cy)
            (self.sx_hint, self.sy_hint),
            roi,                   # ROI image
            roi_sub,
            (x0, y0, x1, y1),      # ROI box
        ) = self.tracker.process_frame(gray,self.cx, self.cy,self.sx_hint,self.sy_hint,self.miss,)
    
        if roi_sub.size == 0 or not np.any(roi_sub):
            self.beam_dot.setData([], [])
            return
# ============================================================
# DEFINE BEAM CENTER FROM 1-D PROJECTIONS
# ============================================================
        # px = roi_sub.sum(axis=0)
        # py = roi_sub.sum(axis=1)
        # ============================================================
        # DEFINE BEAM CENTER FROM ROTATED SLIT PROJECTIONS
        # ============================================================
        theta_slit_deg = 0.0
        theta_slit = np.deg2rad(theta_slit_deg)   # <-- put your slit angle here
        
        px, py = rotated_profiles_fast(roi_sub, theta_slit)
        # light smoothing is OK but optional
        px_s = gaussian_filter1d(px, 2.0)
        py_s = gaussian_filter1d(py, 2.0)
    
        ix_peak = int(np.argmax(px_s))
        iy_peak = int(np.argmax(py_s))
        
# ============================================================
# DISPLAY TRANSFORM (10-bit → 8-bit)
# ============================================================
        roi_u8 = np.clip(roi_sub, 0, 1023) #/ 4.0
        roi_u8 = roi_u8.astype(np.uint8)
    
        p2, p98 = np.percentile(roi_u8, (2, 98))
        if p98 <= p2:
            p98 = p2 + 1
    
        self.img_item.setLevels((p2, p98))
        
        roi_display = rotate_image(roi_u8, theta_slit_deg)
        self.img_item.setImage(roi_display, autoLevels=False)

        # self.img_item.setImage(roi_u8, autoLevels=False)
    
        h, w = roi_u8.shape
    
        # ---- lock ViewBox to ROI pixels ----
        if not hasattr(self, "_vb_roi_shape") or self._vb_roi_shape != (w, h):
            vb = self.img_view.getViewBox()
            vb.enableAutoRange(x=False, y=False)
            vb.setLimits(xMin=0, xMax=w, yMin=0, yMax=h)
            vb.setXRange(0, w, padding=0)
            vb.setYRange(0, h, padding=0)
            self._vb_roi_shape = (w, h)

# ============================================================
# Print acquisition stats every 100 frames (~2 seconds @ 50 Hz)
# ============================================================
        if self.frame_idx >= self.window_size and self.frame_idx % self.window_size == 0:
            N = len(self.FX)
            T = N * self.dt
        
            print(
                f"[Acquisition] "
                f"N = {N} samples, "
                f"T = {T:.2f} s, "
                f"Δf = {1.0 / T:.4f} Hz"
            )
# ============================================================
# CYAN DOT = PROJECTION PEAK (ALIGNS WITH GAUSSIAN)
# ============================================================
        # Beam center in ROI pixel coordinates (camera frame)
        cx_roi = bx - x0
        cy_roi = by - y0
        
        self.beam_dot.setData([cx_roi], [cy_roi])
        # self.beam_dot.setData([ix_peak], [iy_peak])
        # print("Peak (proj):", ix_peak, iy_peak, "ROI:", w, h)
    
        # ---- Recording ----
        if self.record_video:
            if not self.rec_started:
                self.recorder.start(roi_u8.shape)
                self.rec_started = True
            self.recorder.write(roi_u8)
            
        self.X.append(self.frame_idx)
        self.INT.append(inten)
        self.BX.append(bx)
        self.BY.append(by)
        self.FX.append(fx)
        self.FY.append(fy)
        self.DX.append(dx)
        self.DY.append(dy)
        self.fft_fx.push(fx)
        self.fft_fy.push(fy)
        self.fft_bx.push(bx)
        self.fft_by.push(by)
        self.int_line.setData(self.X, self.INT)
        self.bx_line.setData(self.X, self.BX)
        self.by_line.setData(self.X, self.BY)
        # self.dx_line.setData(self.X, self.DX)
        # self.dy_line.setData(self.X, self.DY)
        self.fx_line.setData(self.X, self.FX)
        self.fy_line.setData(self.X, self.FY)


        if self.record_csv:
            self.live_records.append(
                [self.frame_idx, bx, by, fx, fy, dx, dy, inten]
            )

        xx = np.arange(len(px_s)) - ix_peak
        yy = np.arange(len(py_s)) - iy_peak
        
        self.line_px.setData(xx, px_s)
        self.line_py.setData(py_s, yy)

        # ---- FFT (block update per window) ----
        if self.frame_idx >= self.window_size and self.frame_idx % self.window_size == 0:
        
            def update_fft(engine, line, peaks_item):
                f, psd = engine.compute()
                if f is None:
                    return
        
                line.setData(f, psd)
        
                peaks, _ = find_peaks(psd, prominence=0.15 * np.max(psd))
                peaks_item.setData(f[peaks], psd[peaks])
        
                # IMPORTANT: reset so next window is independent
                engine.reset()
        
            update_fft(self.fft_fx, self.fft_fx_line, self.fft_fx_peaks)
            update_fft(self.fft_fy, self.fft_fy_line, self.fft_fy_peaks)
            update_fft(self.fft_bx, self.fft_bx_line, self.fft_bx_peaks)
            update_fft(self.fft_by, self.fft_by_line, self.fft_by_peaks)
        
            N = len(self.FX)
            T = N * self.dt
            df = self.fs / self.window_size   # ✅ correct Δf for block FFT
        
            self.lbl_acq.setText(
                f'FPS={self.fs:.1f} | '
                f'FWHM={fx:.1f}×{fy:.1f} µm | '
                f'N={N} | T={T:.1f}s | Δf={df:.4f} Hz'
            )
        self.frame_idx += 1
# ============================================================
# Stop , Recording, Camera, BPM and Tk windows if running
# ============================================================
    def stop(self):
        print("[LiveDashboardQt] Stopping…")

        if self.stop_requested:
            return
        self.stop_requested = True

        self.timer.stop()

        if self.recorder:
            print("[LiveDashboardQt] Stopping recorder…")
            try:
                self.recorder.stop()
                print("[LiveDashboardQt] Stopped recorder…")
            except Exception as e:
                print("Recorder stop error:", e)

        if hasattr(self.tracker, "bpm") and self.tracker.bpm is not None:
            try:
                print("[LiveDashboardQt] Closing BPM window...")
                self.tracker.bpm.stop()
                print("[LiveDashboardQt] Closed BPM window...")
            except Exception as e:
                print("[LiveDashboardQt] BPM stop error:", e)

        # Thorlabs SDK explicit disposal
        try:
            if hasattr(cam, "sdk"):
                print("[LiveDashboardQt] Closing Thorlabs Camera...")
                cam.sdk.dispose()
                print("[LiveDashboardQt] Closed Thorlabs Camera...")
        except Exception:
            pass
            
        if self.record_csv and len(self.live_records) > 0 and self.csv_path:
            df = pd.DataFrame(self.live_records, columns=[
                "frame","beam_x","beam_y","FWHM_x_um","FWHM_y_um",
                "div_x","div_y","intensity"
            ])
            df.to_csv(self.csv_path, index=False)
            print(f"[LiveDashboardQt] CSV saved → {self.csv_path}")
        print("[LiveDashboardQt] stopped.")
        self.close()

# ========================================================================
# Tkinter canvas that displays images
# ========================================================================
class LiveViewCanvas(tk.Canvas):
    def __init__(self, parent, image_queue):
        self.image_queue = image_queue
        self._image_width = 0
        self._image_height = 0
        super().__init__(parent)
        self.pack()
        self._get_image()

    def _get_image(self):
        try:
            image = self.image_queue.get_nowait()
            self._image = ImageTk.PhotoImage(master=self, image=image)

            if (self._image.width() != self._image_width) or (self._image.height() != self._image_height):
                self._image_width = self._image.width()
                self._image_height = self._image.height()
                self.config(width=self._image_width, height=self._image_height)

            self.create_image(0, 0, image=self._image, anchor="nw")

        except queue.Empty:
            pass

        self.after(10, self._get_image)

# ========================================================================
# Thread that reads camera frames to a queue
# ========================================================================
class ImageAcquisitionThread(threading.Thread):
    def __init__(self, camera):
        super().__init__()
        self._camera = camera

        if camera.camera_sensor_type != SENSOR_TYPE.BAYER:
            self._is_color = False
        else:
            self._mono_to_color_sdk = MonoToColorProcessorSDK()
            self._image_width = camera.image_width_pixels
            self._image_height = camera.image_height_pixels
            self._mono_to_color_processor = self._mono_to_color_sdk.create_mono_to_color_processor(
                SENSOR_TYPE.BAYER,
                camera.color_filter_array_phase,
                camera.get_color_correction_matrix(),
                camera.get_default_white_balance_matrix(),
                camera.bit_depth
            )
            self._is_color = True

        self._bit_depth = camera.bit_depth
        self._camera.image_poll_timeout_ms = 0
        self._image_queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()

    def get_output_queue(self):
        return self._image_queue

    def stop(self):
        self._stop_event.set()

    def _get_color_image(self, frame):
        w = frame.image_buffer.shape[1]
        h = frame.image_buffer.shape[0]

        if (w != self._image_width) or (h != self._image_height):
            self._image_width = w
            self._image_height = h

        color_data = self._mono_to_color_processor.transform_to_24(
            frame.image_buffer, self._image_width, self._image_height
        )
        color_data = color_data.reshape(self._image_height, self._image_width, 3)
        return Image.fromarray(color_data, mode="RGB")

    def _get_mono_image(self, frame):
        img = frame.image_buffer >> (self._bit_depth - 8)
        return Image.fromarray(img)

    def run(self):
        while not self._stop_event.is_set():
            try:
                frame = self._camera.get_pending_frame_or_null()
                if frame is not None:
                    img = (
                        self._get_color_image(frame)
                        if self._is_color else
                        self._get_mono_image(frame)
                    )
                    self._image_queue.put_nowait(img)

            except queue.Full:
                pass

        if self._is_color:
            self._mono_to_color_processor.dispose()
            self._mono_to_color_sdk.dispose()

# ========================================================================
# BPM CLASS
# ========================================================================
class BPM:
    """
    Encapsulates the Thorlabs TkInter live viewer.
    Allows:
        bpm = BPM()
        await bpm.run_async()
    """

    def __init__(self, camera: ThorCamera):
        self.camera = camera.camera      # <-- Use existing camera handle
        self.sdk = camera.sdk            # <-- Use existing SDK
        self.acq = None
        self.root = None
        self._thread = None
        self._running = False

    # -------- Internal Setup --------
    def _start(self):
        """Runs the TkInter live view (blocking), so must be threaded."""

        self.acq = ImageAcquisitionThread(self.camera)

        # Create Tk window
        self.root = tk.Tk()
        self.root.title(self.camera.name)

        LiveViewCanvas(
            parent=self.root,
            image_queue=self.acq.get_output_queue()
        )

        # Start acquisition
        self.acq.start()

        # Blocking loop
        self.root.mainloop()

        # Cleanup
        self.acq.stop()
        self.acq.join()
        self.camera.dispose()
        self.sdk.dispose()

        self._running = False

    # -------- Public Async Method --------
    async def run_async(self):
        """Runs the TkInter UI in a background thread without blocking asyncio."""
        if self._running:
            return

        self._running = True

        self._thread = threading.Thread(target=self._start, daemon=True)
        self._thread.start()

        await asyncio.sleep(0.1)

    def stop(self):
        """Stops the live viewer."""
        if self.root:
            self.root.quit()

# ========================================================================
# Camera Info
# ========================================================================
class CameraApp:
    def print_camera_info(self, camera):
        print("=== Camera Information ===")
        print("Model:", camera.model)

        try:
            print("Serial number:", camera.serial_number)
        except Exception:
            print("Serial number: not available")

        print("Sensor resolution:",
              camera.sensor_width_pixels,
              "x",
              camera.sensor_height_pixels)
        print("sensor_pixel_width_um:", camera.sensor_pixel_width_um)
        print("sensor_pixel_height_um:", camera.sensor_pixel_height_um)
        print("Exposure time (µs):", camera.exposure_time_us)
        print("Frame time (µs):", camera.frame_time_us)
        print(f"Frame per Sec (FPS): {1e6 / camera.frame_time_us:.1f}")
        print("Bit depth:", camera.bit_depth)
        print("Operation mode:", camera.operation_mode)
        print("==========================")
# ========================================================================
# Function Call
# ========================================================================
def bpm_camera(
    csv_path: str | None = None,
    record_path: str | None = None,
    window_size: int | None = None,
):
    """
    Launch BioXAS live beam diagnostics.

    If window_size, csv_path, or record_path are None,
    the user will be prompted interactively.
    """

    # ======================================================
    # Ask user for FFT window size if not provided
    # ======================================================
    if window_size is None:
        while True:
            try:
                window_size = int(
                    input("Enter FFT window size (samples, e.g. 20 or 100): ")
                )
                if window_size < 10:
                    print("window_size must be >= 10")
                    continue
                break
            except ValueError:
                print("Please enter a valid integer.")

    print(f"[Info] Using FFT window_size = {window_size}")

    # ======================================================
    # Ask user for CSV path if not provided
    # ======================================================
    if csv_path is None:
        while True:
            csv_path = input(
                "Enter CSV output path (press ENTER to disable CSV recording): "
            ).strip()

            if csv_path == "":
                csv_path = None
                print("[Info] CSV recording disabled.")
                break

            if not csv_path.lower().endswith(".csv"):
                print("CSV path must end with '.csv'")
                continue

            csv_dir = os.path.dirname(csv_path)
            if csv_dir and not os.path.isdir(csv_dir):
                print(f"Directory does not exist: {csv_dir}")
                continue

            print(f"[Info] CSV will be saved to: {csv_path}")
            break

    # ======================================================
    # Ask user for ROI recording path if not provided
    # ======================================================
    if record_path is None:
        while True:
            record_path = input(
                "Enter ROI video path (press ENTER to disable recording): "
            ).strip()

            if record_path == "":
                record_path = None
                print("[Info] ROI video recording disabled.")
                break

            if not record_path.lower().endswith((".mp4", ".avi")):
                print("Recording path must end with '.mp4' or '.avi'")
                continue

            rec_dir = os.path.dirname(record_path)
            if rec_dir and not os.path.isdir(rec_dir):
                print(f"Directory does not exist: {rec_dir}")
                continue

            print(f"[Info] ROI video will be saved to: {record_path}")
            break

    # ======================================================
    # Camera + Tracker
    # ======================================================
    cam = ThorCamera()
    roi = ROIManager()
    analyzer = BeamAnalyzer(method="moments")

    tracker = BeamTracker(
        camera=cam,
        analyzer=analyzer,
        roi_mgr=roi,
    )

    # ======================================================
    # BPM (Tkinter) viewer
    # ======================================================
    bpm = BPM(camera=cam)
    tracker.bpm = bpm
    threading.Thread(target=bpm._start, daemon=True).start()

    # ======================================================
    # Qt Application
    # ======================================================
    qt_app = QApplication.instance()
    if qt_app is None:
        qt_app = QApplication(sys.argv)

    dashboard = LiveDashboardQt(
        tracker,
        csv_path=csv_path,
        record_path=record_path,
        window_size=window_size,
    )

    # ======================================================
    # Camera info
    # ======================================================
    info = CameraApp()
    info.print_camera_info(cam.camera)

    # ======================================================
    # Show dashboard
    # ======================================================
    dashboard.show()
    qt_app.exec()

    return dashboard
