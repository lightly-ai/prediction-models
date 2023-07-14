# Custom Prediction Models

Code to create Lightly predictions with ONNX object detection models.


## Installation
Install dependencies with:
```
pip install -r requirements.txt
```


## Usage

**Overview**

1. Get an ONNX model to use for predictions. See [here](#models) for supported models.
2. Create a [config](#config) file to specify the input and output format of the ONNX
    model. This is required for doing the right pre and postprocessing.
3. Prepare a directory with images for which you want to create predictions.
4. Run the prediction script with the config file.
5. Finally, images with the bounding boxes overlayed are saved in the output directory.
    If the bounding boxes are as expected, it means that the model and config were used
    correctly.

First, Download or create ONNX model weights for the model you want to use. The
supported models are listed [here](#models). Any ONNX model that has the same input and
output nodes as one of the supported models can be used.

Next, copy one of the example configs in the [configs](configs) directory and adapt it
to your needs. See [here](#config) how to do this. In most cases you only need to change
two settings:
* The `path` to the ONNX model weights.
* The `schema` of the prediction task. This must contain all categories from the model.

Next, prepare a directory with images for which you want to create predictions. The
images can be in any format supported by PIL. If you don't have any images, the script
will automatically download some example COCO images for you.
> Note: The script is not optimized for large datasets and loads one image at a time.

Finally, run the prediction script with:
```
python predict.py \
    --image-dir "path/to/images" \
    --output-dir "outputs" \
    --model-config "configs/my_config.json"
```

Relative paths are resolved relative to the current working directory.

After running the script, the output directory will contain Lightly prediction files and
plots of the original images with the Lightly predictions. A new `output` directory is
created if `--output-dir` is omitted. Every run of the script will create a new
subdirectory with the current timestamp in the output directory.


### Models

Any ONNX model that has the same input and output nodes as one of the supported models
can be used. Note that Lightly automatically infers the input image size from the model.


#### `WongKinYiu/yolov7`

Follow the steps here to export the model to ONNX: https://github.com/WongKinYiu/yolov7#export

We recommend to export including NMS.

Default config: [configs/wongkinyiu_yolov7.json](configs/wongkinyiu_yolov7.json)


**Inputs**
```
images:
    Array with shape (1, 3, H, W).
```

**Outputs**
```
output:
    Array with shape (num_predictions, 7). Dim 1 of array contains
    [batch_index, x0, x1, y0, y1, confidence, class_index].
```

#### `ultralytics/ultralytics:yolov8`

Follow the steps here to export the model to ONNX: https://github.com/ultralytics/ultralytics#python

The exported model does **not** include NMS. Remember to add nms postprocessing to the
[config](#config) file!

Default config: [configs/ultralytics_ultralytics_yolov8.json](configs/ultralytics_ultralytics_yolov8.json)

**Inputs**
```
images:
    Array with shape (1, 3, H, W).
```
**Outputs**
```
output0:
    Array with shape (1, 4 + num_categories, num_predictions). Dim 1 of array
    contains [cx, cy, w, h, confidence_category_0, ..., confidence_category_n].
```


### Config

The config file contains all the information needed to run the prediction script. The
configuration flags are explained below:
```json5
{
    // required
    // Prediction model. Currently supports:
    // - WongKinYiu/yolov7
    // - ultralytics/ultralytics:yolov8
    // This flag is required for Lightly to provide images in the right format to the
    // model and convert the model output to Lightly predictions.
    "model": "WongKinYiu/yolov7",

    // required
    // Format of the model. Currently only supports ONNX.
    "format": "onnx",

    // required
    // Path to the ONNX model weights. Currently only supports a local path. Relative 
    // paths are resolved relative to the current working directory.
    "path": "models/wongkinyiu_yolov7_tiny.onnx",

    // required
    // Lightly prediction task definition. Follows the same format as Lightly
    // predictions uploaded the a datasource, see https://docs.lightly.ai/docs/prediction-format
    "task": {
        // Task name, will be used for selection and displayed on the Lightly Platform.
        "name": "my-prediction-task",

        // Prediction schema described in https://docs.lightly.ai/docs/prediction-format#prediction-schema
        // Must contain all categories from the model.
        "schema": {
            "task_type": "object-detection",
            "categories": [
                {
                    "id": 0,
                    "name": "person"
                },
                {
                    "id": 1,
                    "name": "bicycle"
                },
                ...
            ]
        }
    }

    // optional
    // Preprocessing steps applied to the image before passing it to the model.
    "preprocess": {
        // Normalization parameters. Images are converted to RGB and scaled to
        // [0, 1] before normalization. By default no normalization is applied.
        "normalize": {
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0]
        },
    }

    // optional
    // Postprocessing steps applied to the model outputs before converting them to
    // Lightly predictions. By default no postprocessing is applied.
    "postprocess": {

        // Keeps only predictions with a confidence score above the threshold.
        // By default no threshold is applied.
        "confidence_threshold": 0.0,

        // Non-maximum suppression parameters. By default no NMS is applied.
        // Set this if your ONNX model does not already apply NMS or if you want to
        // be more selective than the model.
        "nms": {
            "iou_threshold": 0.0
        }
    },
}
```