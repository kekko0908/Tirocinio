from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import json
import numpy as np
import cv2
from dotenv import load_dotenv
from ai2thor.controller import Controller


load_dotenv()


@dataclass
class EnvConfig:
    scene: str = "FloorPlan1"
    agent_mode: str = "default"  # in seguito: "arm" o modalità ManipulaTHOR
    width: int = 640
    height: int = 480
    fov: int = 90
    render_depth: bool = True
    render_instance_segmentation: bool = True


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_dir(*parts: str) -> Path:
    p = project_root() / "data"
    for x in parts:
        p = p / x
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_controller(cfg: EnvConfig) -> Controller:
    controller = Controller(
        scene=cfg.scene,
        agentMode=cfg.agent_mode,  # <<< QUI
        width=cfg.width,
        height=cfg.height,
        fieldOfView=cfg.fov,
        renderDepthImage=cfg.render_depth,
        renderInstanceSegmentation=cfg.render_instance_segmentation,
    )
    return controller



def save_rgb(frame_bgr: np.ndarray, name: str) -> Path:
    out = data_dir("frames") / name
    cv2.imwrite(str(out), frame_bgr)
    return out


def get_rgb_bgr(event) -> np.ndarray:
    # ai2thor restituisce RGB; OpenCV vuole BGR
    rgb = event.frame
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def print_ok(msg: str) -> None:
    print(f"[OK] {msg}")


def print_warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def print_err(msg: str) -> None:
    print(f"[ERR] {msg}")
def state_dir():
    return data_dir("state")

def save_json(obj, filename: str):
    p = state_dir() / filename
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return p

def load_json(filename: str):
    p = state_dir() / filename
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)