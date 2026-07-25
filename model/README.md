# TF.js model files

Put your TensorFlow.js model export in this folder.

Expected files (names may vary slightly depending on export):

- `model.json` — model architecture / graph
- `group1-shard1ofN.bin` (and additional shard files if present) — weights

## Keras 3 exports

If `tf.loadLayersModel` fails with `batchInputShape` / `inbound_nodes` / weight-name errors,
re-patch from the original export:

```bash
# keep the converter output as model.json.bak first, then:
python3 ../fix_keras3_model.py model.json --from-backup --inplace
```

That script:

- renames `batch_shape` → `batch_input_shape`
- flattens Keras 3 `DTypePolicy` dtypes
- converts Keras 3 `inbound_nodes` to the legacy TF.js format
- removes unsupported `Rescaling` (use `normalize: "-1-1"` in `app.js` instead)
- renames weight paths TF.js expects (`dense/...`, `depthwise_kernel`)

## How to export

From a Keras / TensorFlow SavedModel:

```bash
pip install tensorflowjs
tensorflowjs_converter --input_format=keras path/to/model.h5 ./model
# or
tensorflowjs_converter --input_format=tf_saved_model path/to/saved_model ./model
```

## Git LFS

Large weight shards should be tracked with Git LFS. From the repo root:

```bash
git lfs install
git lfs track "Anya_WVA_AD/model/*.bin"
git add Anya_WVA_AD/.gitattributes Anya_WVA_AD/model/
```

Update `../labels.json` so the class order matches your model's output units
(this model outputs **90** classes).
If you trained with 224×224 inputs and MobileNet `[-1, 1]` normalization,
keep `normalize: "-1-1"` in `../app.js` (`MODEL_CONFIG`).
