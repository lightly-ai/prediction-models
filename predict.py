import json
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeAlias
from urllib import request

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from numpy.typing import NDArray
from onnxruntime import InferenceSession
from PIL import Image
from PIL.Image import Image as PILImage
from torchvision import ops
from torchvision.transforms import Pad
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument(
    "--image-dir",
    type=Path,
    help=(
        "Path to directory with images. If not specified, example images will be "
        "downloaded. (default: None)"
    ),
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default="outputs",
    help="Directory where outputs are stored. (default: outputs)",
)
parser.add_argument(
    "--model-config",
    type=Path,
    default="configs/wongkinyiu_yolov7.json",
    help="Path to model config. (default: configs/wongkinyiu_yolov7.json)",
)


ModelConfig: TypeAlias = dict[str, Any]
PredictionSingleton: TypeAlias = dict[str, Any]
Prediction: TypeAlias = dict[str, str | list[PredictionSingleton]]


def main(image_dir: Path | None, output_dir: Path, model_config: Path):
    config = _load_config(path=model_config)
    image_dir = _prepare_image_dir(image_dir=image_dir)
    image_paths = _get_image_paths(image_dir=image_dir)
    print(f"Found {len(image_paths)} images in '{image_dir.resolve()}'.")
    model = PREDICTION_MODEL_REGISTRY[config["model"]](config=config)
    category_names = _get_category_names(config=config)
    output_dir = _prepare_output_dir(output_dir=output_dir)
    _save_model_config(
        output_dir=output_dir,
        config=config,
    )
    _save_lightly_task_schema(
        output_dir=output_dir,
        config=config,
    )
    print("Creating predictions...")
    for image_path in tqdm(image_paths):
        image = Image.open(image_path).convert("RGB")
        predictions = model.predict(image=image)
        filename = str(image_path.relative_to(image_dir))
        _plot_lightly_predictions(
            output_path=(output_dir / "lightly_plots" / filename).with_suffix(".png"),
            image=image,
            predictions=predictions,
            category_names=category_names,
        )
        _save_lightly_predictions(
            output_dir=output_dir,
            config=config,
            filename=filename,
            predictions=predictions,
        )


def _load_config(path: Path) -> ModelConfig:
    print("Loading model config...")
    with path.open() as f:
        return json.load(f)


def _prepare_image_dir(image_dir: Path | None) -> Path:
    if image_dir is None:
        image_dir = Path("images")
        _download_images(image_dir=image_dir)
    return image_dir


def _get_category_names(config: ModelConfig) -> list[str]:
    categories = config["task"]["schema"]["categories"]
    categories = sorted(categories, key=lambda x: x["id"])
    assert [category["id"] for category in categories] == list(range(len(categories)))
    return [category["name"] for category in categories]


def _prepare_output_dir(output_dir: Path) -> Path:
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = output_dir / now
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir


def _download_images(image_dir: Path):
    image_dir.mkdir(exist_ok=True, parents=True)
    urls = [
        "http://farm6.staticflickr.com/5088/5366642189_ecc1af26c1_z.jpg",
        "http://farm8.staticflickr.com/7204/6982099079_d152fb67b1_z.jpg",
        "http://farm8.staticflickr.com/7373/9108705512_bc5fd66b03_z.jpg",
    ]
    for url in urls:
        image_path = image_dir / Path(url).name
        if not image_path.is_file():
            print(f"Downloading '{url}' to '{image_path.resolve()}'...")
            request.urlretrieve(url, image_path)


def _get_image_paths(image_dir: Path):
    """Get paths for all images in directory."""
    return sorted(
        p for p in image_dir.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    )


def _pil_resize(image: PILImage, size: tuple[int, int]) -> tuple[PILImage, float]:
    """Resize image to size in (width, height) format while maintaining aspect ratio.

    The image is resized to the largest size that fits within the specified size.
    """
    img_width, img_height = image.width, image.height
    ratio = min(size[0] / img_width, size[1] / img_height)
    new_width = int(img_width * ratio)
    new_height = int(img_height * ratio)
    return image.resize((new_width, new_height), resample=Image.BICUBIC), ratio


def _pil_pad(
    image: PILImage, size: tuple[int, int]
) -> tuple[PILImage, tuple[int, int, int, int]]:
    """Pad image to size in (width, height) format."""
    img_width, img_height = image.width, image.height
    delta_w = size[0] - img_width
    delta_h = size[1] - img_height
    padding = (
        delta_w // 2,
        delta_h // 2,
        delta_w - (delta_w // 2),
        delta_h - (delta_h // 2),
    )
    transform = Pad(padding=padding, fill=(127, 127, 127), padding_mode="constant")
    return transform(image), padding


def _np_normalize(
    image_rgb: NDArray[np.float32],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> NDArray[np.float32]:
    """Normalize image with mean and std."""
    mean_arr = np.array(mean, dtype=np.float32).reshape((3, 1, 1))
    std_arr = np.array(std, dtype=np.float32).reshape((3, 1, 1))
    return (image_rgb - mean_arr) / std_arr


def _np_nms(
    xyxy: NDArray[np.float32],
    scores: NDArray[np.float32],
    categories: NDArray[np.float32],
    iou_threshold: float,
) -> NDArray[np.int_]:
    """Apply non-maximum suppression.

    Returns indices of detections to keep.
    """
    indices = ops.batched_nms(
        boxes=torch.from_numpy(xyxy),
        scores=torch.from_numpy(scores),
        idxs=torch.from_numpy(categories),
        iou_threshold=iou_threshold,
    )
    return indices.numpy()


def _rescale_bbox(
    xyxy: NDArray[np.float32],
    original_size: tuple[int, int],
    resize_ratio: float,
    padding: tuple[int, int, int, int],
) -> NDArray[np.float32]:
    """Convert bounding boxes to original image coordinates.

    Returns bounding boxes in (x0, y0, x1, y1) format.
    """
    # Remove padding
    xyxy = xyxy - np.array(
        [padding[0], padding[1], padding[0], padding[1]], dtype=np.float32
    )
    # Rescale bbox to original image size
    xyxy /= resize_ratio
    # Clip bbox to original image size
    xyxy[:, 0] = np.clip(xyxy[:, 0], 0, original_size[0])
    xyxy[:, 1] = np.clip(xyxy[:, 1], 0, original_size[1])
    xyxy[:, 2] = np.clip(xyxy[:, 2], 0, original_size[0])
    xyxy[:, 3] = np.clip(xyxy[:, 3], 0, original_size[1])
    return xyxy


def _plot_lightly_predictions(
    output_path: Path,
    image: PILImage,
    predictions: list[PredictionSingleton],
    category_names: list[str],
) -> None:
    """Plot predictions on image."""
    output_path.parent.mkdir(exist_ok=True, parents=True)
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i % 20) for i in range(len(category_names))]
    plt.subplots(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()
    for prediction in predictions:
        bbox = prediction["bbox"]
        category_id = prediction["category_id"]
        score = prediction["score"]
        color = colors[category_id % 20]
        x, y, w, h = bbox
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                fill=False,
                facecolor="none",
                edgecolor=color,
                linewidth=1,
            )
        )
        ax.text(
            x,
            y,
            f"{category_names[category_id]} {score:.4f}",
            bbox=dict(facecolor=color, alpha=0.5),
            fontsize=12,
            color="white",
        )
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
    plt.close()


def _save_lightly_predictions(
    output_dir: Path,
    config: ModelConfig,
    filename: str,
    predictions: list[PredictionSingleton],
) -> None:
    """Save predictions to file."""
    task_name = config["task"]["name"]
    output_path = (
        output_dir / "lightly_predictions" / task_name / filename
    ).with_suffix(".json")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with output_path.open("w") as f:
        json.dump(
            {
                "filename": filename,
                "predictions": predictions,
            },
            f,
            indent=4,
        )


def _save_model_config(
    output_dir: Path,
    config: ModelConfig,
) -> None:
    """Save model config to file."""
    output_path = output_dir / "model_config.json"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with output_path.open("w") as f:
        json.dump(config, f, indent=4)


def _save_lightly_task_schema(
    output_dir: Path,
    config: ModelConfig,
) -> None:
    """Save task schema to file."""
    task_name = config["task"]["name"]
    output_path = output_dir / "lightly_predictions" / task_name / "schema.json"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with output_path.open("w") as f:
        json.dump(config["task"]["schema"], f, indent=4)


@dataclass()
class PreprocessedImage:
    inputs: dict[str, NDArray[np.float32]]
    original_size: tuple[int, int]
    resize_ratio: float
    padding: tuple[int, int, int, int]


def _preprocess(
    image_rgb: PILImage,
    input_name: str,
    input_size: tuple[int, int],
    normalize: dict[str, Any] | None,
) -> PreprocessedImage:
    """Preprocess image for object detection model inference."""
    image, resize_ratio = _pil_resize(image=image_rgb, size=input_size)
    image, padding = _pil_pad(image=image, size=input_size)

    # (C, H, W)
    image_arr = np.array(image, dtype=np.float32).transpose(2, 0, 1)
    # Convert pixel values from [0, 255] to [0, 1]
    image_arr /= 255.0
    if normalize is not None:
        image_arr = _np_normalize(
            image_rgb=image_arr,
            mean=normalize["mean"],
            std=normalize["std"],
        )
    # (1, C, H, W)
    image_arr = np.expand_dims(image_arr, axis=0)
    return PreprocessedImage(
        inputs={input_name: image_arr},
        original_size=(image_rgb.width, image_rgb.height),
        resize_ratio=resize_ratio,
        padding=padding,
    )


def _postprocess(
    bbox: NDArray[np.float32],
    bbox_format: str,
    score: NDArray[np.float32],
    category: NDArray[np.float32],
    confidence: NDArray[np.float32] | None,
    preprocessed_image: PreprocessedImage,
    confidence_threshold: float | None,
    iou_threshold: float | None,
) -> list[PredictionSingleton]:
    """Postprocess object detection model outputs and convert to lightly predictions."""
    xyxy = ops.box_convert(
        torch.from_numpy(bbox), in_fmt=bbox_format, out_fmt="xyxy"
    ).numpy()

    # Apply confidence threshold
    if confidence_threshold is not None:
        mask = score > confidence_threshold
        xyxy = xyxy[mask]
        category = category[mask]
        score = score[mask]
        if confidence is not None:
            confidence = confidence[mask]

    # Apply nms
    if iou_threshold is not None:
        mask = _np_nms(
            xyxy=xyxy,
            scores=score,
            categories=category,
            iou_threshold=iou_threshold,
        )
        xyxy = xyxy[mask]
        category = category[mask]
        score = score[mask]

    # Convert to original image coordinates
    xyxy = _rescale_bbox(
        xyxy=xyxy,
        original_size=preprocessed_image.original_size,
        resize_ratio=preprocessed_image.resize_ratio,
        padding=preprocessed_image.padding,
    )

    # Convert bounding boxes to xywh
    xywh = ops.box_convert(
        boxes=torch.from_numpy(xyxy), in_fmt="xyxy", out_fmt="xywh"
    ).numpy()

    # Get probabilities
    probability = (
        confidence / confidence.sum(axis=1, keepdims=True)
        if confidence is not None
        else None
    )

    # Convert to lightly predictions
    predictions = []
    for i in range(len(xywh)):
        prediction = {
            "bbox": xywh[i].astype(int).tolist(),
            "category_id": int(category[i]),
            "score": float(score[i]),
        }
        if probability is not None:
            prediction["probabilities"] = probability[i].astype(float).tolist()
        predictions.append(prediction)
    return predictions


class PredictionModel:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        print("Loading ONNX model...")
        self.session = InferenceSession(
            path_or_bytes=config["path"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        print("Model inputs:")
        for i in self.session.get_inputs():
            print(f"  {i}")
        print("Model outputs:")
        for o in self.session.get_outputs():
            print(f"  {o}")

        self.normalize = self.config.get("preprocess", {}).get("normalize")
        self.confidence_threshold = self.config.get("postprocess", {}).get(
            "confidence_threshold"
        )
        self.iou_threshold = (
            self.config.get("postprocess", {}).get("nms", {}).get("iou_threshold")
        )

    def predict(self, image: PILImage) -> list[PredictionSingleton]:
        proprocessed_image = self.preprocess(image_rgb=image)
        outputs = self.session.run(None, proprocessed_image.inputs)
        return self.postprocess(outputs=outputs, preprocessed_image=proprocessed_image)

    def preprocess(self, image_rgb: PILImage) -> PreprocessedImage:
        raise NotImplementedError

    def postprocess(
        self, outputs: list[NDArray[np.float32]], preprocessed_image: PreprocessedImage
    ) -> list[PredictionSingleton]:
        raise NotImplementedError


class WongKinYiuYOLOv7(PredictionModel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config=config)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = self.session.get_inputs()[0].shape[2:4]

    def preprocess(self, image_rgb: PILImage) -> PreprocessedImage:
        return _preprocess(
            image_rgb=image_rgb,
            input_name=self.input_name,
            input_size=self.input_size,
            normalize=self.normalize,
        )

    def postprocess(
        self, outputs: list[NDArray[np.float32]], preprocessed_image: PreprocessedImage
    ) -> list[PredictionSingleton]:
        # (num_predictions, 7)
        # dim 1 contains: [image_index, x0, x1, y0, y1, category_id, confidence]
        out = outputs[0]
        return _postprocess(
            bbox=out[:, 1:5],
            bbox_format="xyxy",
            score=out[:, 6],
            category=out[:, 5],
            confidence=None,
            preprocessed_image=preprocessed_image,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
        )


class UltralyticsUltralyticsYOLOv8(PredictionModel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = self.session.get_inputs()[0].shape[2:4]

    def preprocess(self, image_rgb: PILImage) -> PreprocessedImage:
        return _preprocess(
            image_rgb=image_rgb,
            input_name=self.input_name,
            input_size=self.input_size,
            normalize=self.normalize,
        )

    def postprocess(
        self, outputs: list[NDArray[np.float32]], preprocessed_image: PreprocessedImage
    ) -> list[PredictionSingleton]:
        # (4 + num_categories, num_predictions)
        # dim 0 contains: [cx, cy, w, h, confidence_cls_0, ..., confidence_cls_n]
        out = outputs[0][0]
        # (num_predictions, 4 + num_categories)
        out = out.transpose(1, 0)
        return _postprocess(
            bbox=out[:, :4],
            bbox_format="cxcywh",
            score=out[:, 4:].max(axis=1),
            category=out[:, 4:].argmax(axis=1),
            confidence=out[:, 4:],
            preprocessed_image=preprocessed_image,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
        )


PREDICTION_MODEL_REGISTRY = {
    "WongKinYiu/yolov7": WongKinYiuYOLOv7,
    "ultralytics/ultralytics:yolov8": UltralyticsUltralyticsYOLOv8,
}

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
