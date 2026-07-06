import json
from dataclasses import dataclass


@dataclass
class Phone:
    serial: str
    width: int
    height: int


@dataclass
class Config:
    adb_path: str
    dataset_dir: str
    phones: list[Phone]
    anchors_ratio: dict
    timing: dict
    detector: str = "cv"            # "cv" (classical) or "yolo"
    yolo_model_path: str = ""       # path to trained best.pt / .onnx when detector == "yolo"
    stream: bool = False            # continuous screenrecord frame stream (needs `av`)


def load_config(path):
    with open(path, "r") as f:
        data = json.load(f)
    phones = [Phone(**p) for p in data["phones"]]
    return Config(
        adb_path=data["adb_path"],
        dataset_dir=data["dataset_dir"],
        phones=phones,
        anchors_ratio=data["anchors_ratio"],
        timing=data["timing"],
        detector=data.get("detector", "cv"),
        yolo_model_path=data.get("yolo_model_path", ""),
        stream=data.get("stream", False),
    )


def resolve_point(ratio, phone):
    rx, ry = ratio
    return (round(rx * phone.width), round(ry * phone.height))
