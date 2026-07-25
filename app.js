/**
 * Animal Image Classification — TensorFlow.js client
 *
 * Place your exported TF.js model in ./model/ (model.json + weight shards).
 * Update labels.json so class names match your model's output order.
 * Adjust MODEL_CONFIG if your training pipeline used different sizing/normalization.
 */

const MODEL_CONFIG = {
  modelUrl: "model/model.json",
  labelsUrl: "labels.json",
  // Common defaults for MobileNet / Keras image classifiers.
  imageSize: 224,
  // Model includes MobileNet-style scaling in training (Rescaling removed from JSON).
  // "0-1" => divide by 255; "-1-1" => (x / 127.5) - 1; "none" => raw 0-255
  normalize: "-1-1",
  // Convert every upload to JPEG before preview/classification.
  jpegQuality: 0.92,
};

const statusEl = document.getElementById("status");
const imageInput = document.getElementById("image-input");
const previewImage = document.getElementById("preview-image");
const previewPlaceholder = document.getElementById("preview-placeholder");
const predictionsEl = document.getElementById("predictions");

let model = null;
let labels = [];
let activePreviewUrl = null;

function setStatus(message) {
  statusEl.textContent = message;
}

function renderEmptyPredictions(message = "Waiting for an image") {
  predictionsEl.innerHTML = `
    <li class="prediction empty">
      <span class="prediction-label">${message}</span>
      <span class="prediction-score">—</span>
    </li>
    <li class="prediction empty">
      <span class="prediction-label">—</span>
      <span class="prediction-score">—</span>
    </li>
    <li class="prediction empty">
      <span class="prediction-label">—</span>
      <span class="prediction-score">—</span>
    </li>
  `;
}

function renderPredictions(topK) {
  predictionsEl.innerHTML = topK
    .map(({ label, index, probability }) => {
      const pct = probability * 100;
      return `
        <li class="prediction">
          <span class="prediction-label">${label} <span class="prediction-index">(Index: ${index})</span></span>
          <span class="prediction-score">${pct.toFixed(1)}%</span>
          <div class="prediction-bar" aria-hidden="true">
            <span style="width: ${pct.toFixed(1)}%"></span>
          </div>
        </li>
      `;
    })
    .join("");
}

async function loadLabels() {
  const response = await fetch(MODEL_CONFIG.labelsUrl);
  if (!response.ok) {
    throw new Error(`Could not load labels.json (${response.status})`);
  }
  const data = await response.json();
  if (!Array.isArray(data) || data.length === 0) {
    throw new Error("labels.json must be a non-empty JSON array of class names");
  }
  return data;
}

async function loadModel() {
  // Prefer LayersModel (Keras save); fall back to GraphModel (SavedModel convert).
  try {
    return await tf.loadLayersModel(MODEL_CONFIG.modelUrl);
  } catch (layersError) {
    console.warn("loadLayersModel failed, trying loadGraphModel…", layersError);
    try {
      return await tf.loadGraphModel(MODEL_CONFIG.modelUrl);
    } catch (graphError) {
      console.error("loadGraphModel also failed", graphError);
      throw layersError;
    }
  }
}

/**
 * Decode any browser-supported image and re-encode as JPEG.
 * Matches JPG-trained models more closely (RGB, no alpha, JPEG compression).
 */
function convertToJpeg(file, quality = MODEL_CONFIG.jpegQuality) {
  return new Promise((resolve, reject) => {
    const sourceUrl = URL.createObjectURL(file);
    const image = new Image();

    image.onload = () => {
      try {
        const canvas = document.createElement("canvas");
        canvas.width = image.naturalWidth || image.width;
        canvas.height = image.naturalHeight || image.height;

        if (!canvas.width || !canvas.height) {
          throw new Error("Image has invalid dimensions");
        }

        const ctx = canvas.getContext("2d");
        // JPEG has no alpha — paint a white background first.
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(image, 0, 0);

        canvas.toBlob(
          (blob) => {
            URL.revokeObjectURL(sourceUrl);
            if (!blob) {
              reject(new Error("JPEG conversion failed"));
              return;
            }
            const jpgFile = new File(
              [blob],
              (file.name.replace(/\.[^.]+$/, "") || "upload") + ".jpg",
              { type: "image/jpeg" }
            );
            resolve(jpgFile);
          },
          "image/jpeg",
          quality
        );
      } catch (error) {
        URL.revokeObjectURL(sourceUrl);
        reject(error);
      }
    };

    image.onerror = () => {
      URL.revokeObjectURL(sourceUrl);
      reject(new Error("Could not decode that image for JPEG conversion"));
    };

    image.src = sourceUrl;
  });
}

function preprocess(imageElement) {
  return tf.tidy(() => {
    let tensor = tf.browser.fromPixels(imageElement);
    tensor = tf.image.resizeBilinear(tensor, [
      MODEL_CONFIG.imageSize,
      MODEL_CONFIG.imageSize,
    ]);
    tensor = tensor.toFloat();

    if (MODEL_CONFIG.normalize === "-1-1") {
      tensor = tensor.div(127.5).sub(1);
    } else {
      tensor = tensor.div(255);
    }

    return tensor.expandDims(0);
  });
}

function topKFromScores(scores, k = 3) {
  const indexed = Array.from(scores).map((probability, index) => ({
    index,
    probability,
    label: labels[index] ?? `Class ${index}`,
  }));

  indexed.sort((a, b) => b.probability - a.probability);
  return indexed.slice(0, k);
}

async function classify(imageElement) {
  if (!model) {
    throw new Error("Model is not loaded yet");
  }

  const input = preprocess(imageElement);
  let output;

  try {
    output = model.predict(input);
    // Graph models sometimes return a map or array of tensors.
    if (Array.isArray(output)) {
      output = output[0];
    } else if (output && typeof output === "object" && !(output instanceof tf.Tensor)) {
      const values = Object.values(output);
      output = values[0];
    }

    const scores = await output.data();
    return topKFromScores(scores, 3);
  } finally {
    input.dispose();
    if (output && typeof output.dispose === "function") {
      output.dispose();
    }
  }
}

function showPreview(file) {
  if (activePreviewUrl) {
    URL.revokeObjectURL(activePreviewUrl);
    activePreviewUrl = null;
  }

  const url = URL.createObjectURL(file);
  activePreviewUrl = url;

  previewImage.onload = async () => {
    previewImage.hidden = false;
    previewPlaceholder.hidden = true;

    try {
      setStatus("Classifying…");
      const top = await classify(previewImage);
      renderPredictions(top);
      setStatus("Done (converted to JPG)");
    } catch (error) {
      console.error(error);
      renderEmptyPredictions("Classification failed");
      setStatus(error.message || "Could not classify this image");
    }
  };

  previewImage.onerror = () => {
    setStatus("Could not read the converted JPEG");
  };

  previewImage.src = url;
}

imageInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  // Allow re-selecting the same file later.
  event.target.value = "";

  if (!file) {
    return;
  }
  if (!file.type.startsWith("image/")) {
    setStatus("Please choose an image file");
    return;
  }

  try {
    setStatus("Converting to JPG…");
    const jpgFile = await convertToJpeg(file);
    showPreview(jpgFile);
  } catch (error) {
    console.error(error);
    renderEmptyPredictions("Conversion failed");
    setStatus(error.message || "Could not convert that image to JPG");
  }
});

async function init() {
  renderEmptyPredictions();

  if (typeof tf === "undefined") {
    setStatus("TensorFlow.js failed to load. Check your network connection.");
    return;
  }

  try {
    setStatus("Loading labels…");
    labels = await loadLabels();

    setStatus("Loading model…");
    model = await loadModel();
    // Warm up once so the first user click feels snappy.
    tf.tidy(() => {
      const warm = tf.zeros([1, MODEL_CONFIG.imageSize, MODEL_CONFIG.imageSize, 3]);
      const out = model.predict(warm);
      if (Array.isArray(out)) {
        out.forEach((t) => t.dispose());
      } else if (out && typeof out.dispose === "function") {
        out.dispose();
      } else if (out && typeof out === "object") {
        Object.values(out).forEach((t) => t?.dispose?.());
      }
    });

    setStatus("Ready — upload an animal image");
  } catch (error) {
    console.error(error);
    setStatus(
      "Model not found yet. Add your TF.js files under model/ (see model/README.md)."
    );
  }
}

init();
