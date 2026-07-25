# Animal Image Classification — Anya Dash

Static web app for animal image classification with TensorFlow.js.
Designed to host on **GitHub Pages**.

## Layout

- Banner: **Animal Image Classification** / by Anya Dash
- Left: upload button (camera or gallery on mobile) + image preview
- Right: top 3 class predictions with probabilities

## Files

| Path | Purpose |
|------|---------|
| `index.html` | Page structure |
| `styles.css` | Layout and styling |
| `app.js` | Upload, preprocess, TF.js inference |
| `labels.json` | Class names in model output order |
| `model/` | Your TF.js `model.json` + `.bin` weight shards |

## Add your model

1. Export with `tensorflowjs_converter` into `model/`.
2. Replace `labels.json` with your class list (same order as training).
3. If needed, edit `MODEL_CONFIG` in `app.js` (`imageSize`, `normalize`).
4. Commit weight files with Git LFS (see `model/README.md` and `.gitattributes`).

## Local preview

Serve the folder over HTTP (file:// will block model loading):

```bash
cd Anya_WVA_AD
python3 -m http.server 8080
```

Open http://localhost:8080

## GitHub Pages

1. Push this folder (and LFS objects) to your GitHub repo.
2. Settings → Pages → deploy from branch (`/root` or `/docs`, depending on layout).
3. If the site root is the repo root, the app URL is:

   `https://<user>.github.io/<repo>/Anya_WVA_AD/`

If Pages is published only from `Anya_WVA_AD`, open the site root URL instead.
