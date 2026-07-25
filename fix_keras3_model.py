#!/usr/bin/env python3
"""Patch a Keras 3 → TF.js layers model.json so tf.loadLayersModel can parse it."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


UNSUPPORTED_CONFIG_KEYS = {"optional", "quantization_config"}


def flatten_dtype(value):
    if isinstance(value, dict) and value.get("class_name") == "DTypePolicy":
        return value.get("config", {}).get("name", "float32")
    return value


def keras_tensor_to_legacy(tensor: dict, call_kwargs: dict | None = None) -> list:
    history = tensor["config"]["keras_history"]
    kwargs = {k: v for k, v in (call_kwargs or {}).items() if v is not None}
    return [history[0], history[1], history[2], kwargs]


def convert_inbound_nodes(inbound_nodes):
    """Keras 3 object nodes → legacy nested-array nodes expected by TF.js."""
    if not isinstance(inbound_nodes, list) or not inbound_nodes:
        return inbound_nodes

    # Already legacy: [[["layer", 0, 0, {}], ...], ...]
    first = inbound_nodes[0]
    if isinstance(first, list):
        return inbound_nodes

    converted = []
    for node in inbound_nodes:
        if not isinstance(node, dict) or "args" not in node:
            converted.append(node)
            continue

        args = node.get("args") or []
        kwargs = node.get("kwargs") or {}
        refs = []

        for arg in args:
            if isinstance(arg, dict) and arg.get("class_name") == "__keras_tensor__":
                refs.append(keras_tensor_to_legacy(arg, kwargs))
            elif isinstance(arg, list):
                # e.g. Add: args = [[tensor_a, tensor_b]]
                for item in arg:
                    if isinstance(item, dict) and item.get("class_name") == "__keras_tensor__":
                        refs.append(keras_tensor_to_legacy(item, kwargs))
            # Non-tensor args are ignored for topology wiring.

        converted.append(refs)

    return converted


def normalize_io_list(value):
    """['name', 0, 0] → [['name', 0, 0]]; leave nested lists alone."""
    if (
        isinstance(value, list)
        and len(value) == 3
        and isinstance(value[0], str)
        and all(isinstance(x, int) for x in value[1:])
    ):
        return [value]
    return value


def patch_obj(obj):
    if isinstance(obj, dict):
        if "batch_shape" in obj and "batch_input_shape" not in obj:
            obj["batch_input_shape"] = obj.pop("batch_shape")

        if "dtype" in obj:
            obj["dtype"] = flatten_dtype(obj["dtype"])

        if "inbound_nodes" in obj:
            obj["inbound_nodes"] = convert_inbound_nodes(obj["inbound_nodes"])

        if "input_layers" in obj:
            obj["input_layers"] = normalize_io_list(obj["input_layers"])
        if "output_layers" in obj:
            obj["output_layers"] = normalize_io_list(obj["output_layers"])

        for key in list(obj.keys()):
            if key in UNSUPPORTED_CONFIG_KEYS:
                obj.pop(key, None)
            else:
                patch_obj(obj[key])
    elif isinstance(obj, list):
        for item in obj:
            patch_obj(item)


def remove_rescaling_layers(model_config: dict) -> dict | None:
    """Remove Keras Rescaling from a Sequential stack; return its config if found."""
    cfg = model_config.get("config")
    if not isinstance(cfg, dict):
        return None
    layers = cfg.get("layers")
    if not isinstance(layers, list):
        return None

    rescaling = None
    kept = []
    for layer in layers:
        if layer.get("class_name") == "Rescaling":
            rescaling = layer.get("config")
            continue
        kept.append(layer)
    cfg["layers"] = kept
    return rescaling


def collect_depthwise_names(obj, found: set[str] | None = None) -> set[str]:
    if found is None:
        found = set()
    if isinstance(obj, dict):
        if obj.get("class_name") == "DepthwiseConv2D":
            name = obj.get("config", {}).get("name") or obj.get("name")
            if name:
                found.add(name)
        for value in obj.values():
            collect_depthwise_names(value, found)
    elif isinstance(obj, list):
        for item in obj:
            collect_depthwise_names(item, found)
    return found


def fix_weight_names(data: dict) -> int:
    """Rename Keras 3 / converter weight paths to match TF.js Layers expectations."""
    depthwise = collect_depthwise_names(data)
    changed = 0

    for group in data.get("weightsManifest", []):
        for weight in group.get("weights", []):
            name = weight.get("name")
            if not isinstance(name, str):
                continue
            new_name = name
            if new_name.startswith("sequential/"):
                new_name = new_name[len("sequential/") :]

            parts = new_name.split("/")
            if len(parts) == 2 and parts[0] in depthwise and parts[1] == "kernel":
                new_name = f"{parts[0]}/depthwise_kernel"

            if new_name != name:
                weight["name"] = new_name
                changed += 1

    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_json",
        nargs="?",
        default="model/model.json",
        type=Path,
    )
    parser.add_argument(
        "--from-backup",
        action="store_true",
        help="Read model.json.bak if present (re-patch from original export)",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite model.json (keeps model.json.bak)",
    )
    args = parser.parse_args()

    path: Path = args.model_json
    bak = path.with_suffix(path.suffix + ".bak")
    source = bak if args.from_backup and bak.exists() else path

    original = json.loads(source.read_text())
    data = copy.deepcopy(original)

    patch_obj(data)

    rescaling = None
    model_config = data.get("modelTopology", {}).get("model_config")
    if isinstance(model_config, dict) and model_config.get("class_name") == "Sequential":
        rescaling = remove_rescaling_layers(model_config)

    renamed = fix_weight_names(data)

    out = path if args.inplace else path.with_name("model.tfjs-patched.json")
    if args.inplace and source != bak and not bak.exists():
        bak.write_text(json.dumps(original))
        print(f"Wrote backup {bak}")
    elif args.inplace and source == bak:
        print(f"Re-patched from backup {bak}")

    out.write_text(json.dumps(data))
    print(f"Wrote {out}")
    print(f"Renamed {renamed} weight entries for TF.js")
    if rescaling:
        scale = rescaling.get("scale")
        offset = rescaling.get("offset")
        print(
            "Removed Rescaling layer. In app.js use normalize matching "
            f"scale={scale}, offset={offset} (MobileNet: normalize='-1-1' on 0-255 pixels)."
        )


if __name__ == "__main__":
    main()
