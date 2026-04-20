#!/usr/bin/env python3
"""
Model conversion pipeline: PyTorch → ONNX → TFLite / Hailo HEF / TensorRT.

Supports:
  - YOLOv8 models from Ultralytics
  - ONNX export with simplification
  - TFLite conversion with INT8 quantization
  - Hailo HEF compilation (requires Hailo Dataflow Compiler)
  - TensorRT engine build (requires tensorrt)

Usage:
    python tools/convert_model.py --model yolov8n.pt --format tflite --quantize int8
    python tools/convert_model.py --model yolov8n.pt --format onnx
    python tools/convert_model.py --model yolov8n.onnx --format hailo --calib-data images/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def export_to_onnx(model_path: str, output_path: str, input_size: int = 640,
                   simplify: bool = True) -> str:
    """Export PyTorch YOLO model to ONNX."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    onnx_path = model.export(
        format="onnx",
        imgsz=input_size,
        simplify=simplify,
        opset=13,
        dynamic=False,
    )
    print(f"ONNX exported: {onnx_path}")

    if output_path and str(onnx_path) != output_path:
        os.rename(onnx_path, output_path)
        onnx_path = output_path
        print(f"Moved to: {output_path}")

    # Verify
    import onnx
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model verified ({onnx_model.graph.input[0].type})")

    return str(onnx_path)


def export_to_tflite(model_path: str, output_path: str, quantize: str = "fp32",
                     calib_data: str | None = None, input_size: int = 640) -> str:
    """Convert ONNX or PyTorch to TFLite with optional quantization."""

    # If .pt, export to ONNX first
    if model_path.endswith(".pt"):
        onnx_path = model_path.replace(".pt", ".onnx")
        model_path = export_to_onnx(model_path, onnx_path, input_size)

    import numpy as np

    try:
        # Method 1: Ultralytics direct export
        from ultralytics import YOLO
        model = YOLO(model_path.replace(".onnx", ".pt"))
        tflite_path = model.export(
            format="tflite",
            imgsz=input_size,
            int8=(quantize == "int8"),
            half=(quantize == "fp16"),
        )
        print(f"TFLite exported via Ultralytics: {tflite_path}")
        return str(tflite_path)

    except Exception:
        pass

    # Method 2: Manual ONNX → TF → TFLite
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf

        print("Converting ONNX → TensorFlow...")
        onnx_model = onnx.load(model_path)
        tf_rep = prepare(onnx_model)
        tf_path = model_path.replace(".onnx", "_tf")
        tf_rep.export_graph(tf_path)

        print("Converting TensorFlow → TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)

        if quantize == "int8" and calib_data:
            def representative_dataset():
                import cv2
                for img_path in sorted(Path(calib_data).glob("*.jpg"))[:100]:
                    img = cv2.imread(str(img_path))
                    img = cv2.resize(img, (input_size, input_size))
                    img = img.astype(np.float32) / 255.0
                    yield [np.expand_dims(img, 0)]

            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            print("INT8 quantization with calibration data")

        elif quantize == "fp16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            print("FP16 quantization")

        tflite_model = converter.convert()
        output = output_path or model_path.replace(".onnx", ".tflite")
        with open(output, "wb") as f:
            f.write(tflite_model)
        print(f"TFLite saved: {output} ({len(tflite_model) / 1e6:.1f} MB)")
        return output

    except ImportError as e:
        print(f"TFLite conversion requires tensorflow and onnx-tf: {e}")
        sys.exit(1)


def export_to_hailo(model_path: str, output_path: str, calib_data: str | None = None,
                    input_size: int = 640) -> str:
    """
    Compile ONNX to Hailo HEF format.

    Requires Hailo Dataflow Compiler (DFC) installed.
    """
    try:
        from hailo_sdk_client import ClientRunner
    except ImportError:
        print(
            "Hailo DFC not installed.\n"
            "Install from: https://hailo.ai/developer-zone/\n"
            "Or use the Hailo Model Zoo CLI: hailo optimize <model>"
        )
        sys.exit(1)

    print("Compiling ONNX → Hailo HEF...")

    runner = ClientRunner(hw_arch="hailo8")
    hn, npz = runner.translate_onnx_model(
        model_path,
        net_name="yolov8n",
        start_node_names=["images"],
        end_node_names=["output0"],
    )

    if calib_data:
        import cv2
        import numpy as np

        calib_images = []
        for img_path in sorted(Path(calib_data).glob("*.jpg"))[:64]:
            img = cv2.imread(str(img_path))
            img = cv2.resize(img, (input_size, input_size))
            img = img.astype(np.float32) / 255.0
            calib_images.append(np.expand_dims(img, 0))

        calib_dataset = np.concatenate(calib_images, axis=0)
        runner.optimize(calib_dataset)
        print("Model optimized with calibration data")

    hef = runner.compile()
    output = output_path or model_path.replace(".onnx", ".hef")
    with open(output, "wb") as f:
        f.write(hef)

    print(f"Hailo HEF saved: {output}")
    return output


def export_to_tensorrt(model_path: str, output_path: str,
                       quantize: str = "fp16", input_size: int = 640) -> str:
    """Build TensorRT engine from ONNX model."""
    try:
        import tensorrt as trt
    except ImportError:
        # Try trtexec CLI
        output = output_path or model_path.replace(".onnx", ".engine")
        precision = "--fp16" if quantize == "fp16" else "--int8" if quantize == "int8" else ""
        cmd = f"trtexec --onnx={model_path} --saveEngine={output} {precision} --workspace=2048"
        print(f"Running: {cmd}")
        os.system(cmd)
        return output

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(model_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            sys.exit(1)

    config = builder.create_builder_config()
    config.max_workspace_size = 2 << 30  # 2 GB

    if quantize == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif quantize == "int8":
        config.set_flag(trt.BuilderFlag.INT8)

    engine = builder.build_engine(network, config)
    output = output_path or model_path.replace(".onnx", ".engine")
    with open(output, "wb") as f:
        f.write(engine.serialize())

    print(f"TensorRT engine saved: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Model conversion pipeline")
    parser.add_argument("--model", "-m", required=True, help="Input model path (.pt or .onnx)")
    parser.add_argument("--format", "-f", required=True,
                        choices=["onnx", "tflite", "hailo", "tensorrt"],
                        help="Target format")
    parser.add_argument("--output", "-o", default=None, help="Output file path")
    parser.add_argument("--quantize", "-q", default="fp32",
                        choices=["fp32", "fp16", "int8"],
                        help="Quantization level")
    parser.add_argument("--calib-data", default=None,
                        help="Calibration image directory for INT8 quantization")
    parser.add_argument("--input-size", type=int, default=640, help="Model input size")

    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Model not found: {args.model}")
        sys.exit(1)

    output = args.output
    if args.format == "onnx":
        export_to_onnx(args.model, output, args.input_size)
    elif args.format == "tflite":
        export_to_tflite(args.model, output, args.quantize, args.calib_data, args.input_size)
    elif args.format == "hailo":
        export_to_hailo(args.model, output, args.calib_data, args.input_size)
    elif args.format == "tensorrt":
        export_to_tensorrt(args.model, output, args.quantize, args.input_size)

    print("Done!")


if __name__ == "__main__":
    main()
