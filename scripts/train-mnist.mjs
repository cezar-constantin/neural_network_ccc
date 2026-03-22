import fs from "node:fs/promises";
import path from "node:path";
import zlib from "node:zlib";

const ROOT = process.cwd();
const DATA_DIR = path.join(ROOT, "data");
const MODEL_DIR = path.join(ROOT, "model");

const TRAIN_IMAGES_FILE = path.join(DATA_DIR, "train-images-idx3-ubyte.gz");
const TRAIN_LABELS_FILE = path.join(DATA_DIR, "train-labels-idx1-ubyte.gz");
const TEST_IMAGES_FILE = path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz");
const TEST_LABELS_FILE = path.join(DATA_DIR, "t10k-labels-idx1-ubyte.gz");

const IMAGE_MAGIC = 2051;
const LABEL_MAGIC = 2049;
const IMAGE_SIZE = 28 * 28;

const HIDDEN_1 = 16;
const HIDDEN_2 = 16;
const OUTPUT = 10;

const TRAIN_LIMIT = 12000;
const TEST_LIMIT = 2000;
const EPOCHS = 12;
const LEARNING_RATE = 0.045;
const L2_REGULARIZATION = 0.00008;

function mulberry32(seed) {
  return function random() {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function createMatrix(rows, cols, random, scale) {
  const matrix = new Array(rows);
  for (let row = 0; row < rows; row += 1) {
    const values = new Float64Array(cols);
    for (let col = 0; col < cols; col += 1) {
      values[col] = (random() * 2 - 1) * scale;
    }
    matrix[row] = values;
  }
  return matrix;
}

function createVector(size) {
  return new Float64Array(size);
}

function relu(value) {
  return value > 0 ? value : 0;
}

function reluDerivative(value) {
  return value > 0 ? 1 : 0;
}

function softmax(logits) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i += 1) {
    if (logits[i] > max) {
      max = logits[i];
    }
  }

  let sum = 0;
  const result = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i += 1) {
    const value = Math.exp(logits[i] - max);
    result[i] = value;
    sum += value;
  }

  for (let i = 0; i < result.length; i += 1) {
    result[i] /= sum;
  }

  return result;
}

function argmax(values) {
  let index = 0;
  for (let i = 1; i < values.length; i += 1) {
    if (values[i] > values[index]) {
      index = i;
    }
  }
  return index;
}

async function ensureDirectory(directory) {
  await fs.mkdir(directory, { recursive: true });
}

async function readGzipFile(filePath) {
  const compressed = await fs.readFile(filePath);
  return zlib.gunzipSync(compressed);
}

async function parseImages(filePath, limit) {
  const buffer = await readGzipFile(filePath);
  const magic = buffer.readUInt32BE(0);
  if (magic !== IMAGE_MAGIC) {
    throw new Error(`Unexpected image magic number in ${filePath}: ${magic}`);
  }

  const count = buffer.readUInt32BE(4);
  const rows = buffer.readUInt32BE(8);
  const cols = buffer.readUInt32BE(12);
  const total = Math.min(count, limit ?? count);

  if (rows * cols !== IMAGE_SIZE) {
    throw new Error(`Unexpected image shape ${rows}x${cols}`);
  }

  const images = new Array(total);
  let offset = 16;
  for (let index = 0; index < total; index += 1) {
    const pixels = new Float64Array(IMAGE_SIZE);
    for (let pixel = 0; pixel < IMAGE_SIZE; pixel += 1) {
      pixels[pixel] = buffer[offset + pixel] / 255;
    }
    offset += IMAGE_SIZE;
    images[index] = pixels;
  }

  return images;
}

async function parseLabels(filePath, limit) {
  const buffer = await readGzipFile(filePath);
  const magic = buffer.readUInt32BE(0);
  if (magic !== LABEL_MAGIC) {
    throw new Error(`Unexpected label magic number in ${filePath}: ${magic}`);
  }

  const count = buffer.readUInt32BE(4);
  const total = Math.min(count, limit ?? count);
  const labels = new Uint8Array(total);
  for (let index = 0; index < total; index += 1) {
    labels[index] = buffer[8 + index];
  }
  return labels;
}

function xavierScale(inputs, outputs) {
  return Math.sqrt(6 / (inputs + outputs));
}

function createNetwork(random) {
  return {
    w1: createMatrix(HIDDEN_1, IMAGE_SIZE, random, xavierScale(IMAGE_SIZE, HIDDEN_1)),
    b1: createVector(HIDDEN_1),
    w2: createMatrix(HIDDEN_2, HIDDEN_1, random, xavierScale(HIDDEN_1, HIDDEN_2)),
    b2: createVector(HIDDEN_2),
    w3: createMatrix(OUTPUT, HIDDEN_2, random, xavierScale(HIDDEN_2, OUTPUT)),
    b3: createVector(OUTPUT),
  };
}

function forward(network, input) {
  const z1 = new Float64Array(HIDDEN_1);
  const a1 = new Float64Array(HIDDEN_1);
  for (let neuron = 0; neuron < HIDDEN_1; neuron += 1) {
    let sum = network.b1[neuron];
    const weights = network.w1[neuron];
    for (let i = 0; i < IMAGE_SIZE; i += 1) {
      sum += weights[i] * input[i];
    }
    z1[neuron] = sum;
    a1[neuron] = relu(sum);
  }

  const z2 = new Float64Array(HIDDEN_2);
  const a2 = new Float64Array(HIDDEN_2);
  for (let neuron = 0; neuron < HIDDEN_2; neuron += 1) {
    let sum = network.b2[neuron];
    const weights = network.w2[neuron];
    for (let i = 0; i < HIDDEN_1; i += 1) {
      sum += weights[i] * a1[i];
    }
    z2[neuron] = sum;
    a2[neuron] = relu(sum);
  }

  const z3 = new Float64Array(OUTPUT);
  for (let neuron = 0; neuron < OUTPUT; neuron += 1) {
    let sum = network.b3[neuron];
    const weights = network.w3[neuron];
    for (let i = 0; i < HIDDEN_2; i += 1) {
      sum += weights[i] * a2[i];
    }
    z3[neuron] = sum;
  }

  const probabilities = softmax(z3);
  return { z1, a1, z2, a2, z3, probabilities };
}

function shuffleIndices(size, random) {
  const indices = new Uint32Array(size);
  for (let i = 0; i < size; i += 1) {
    indices[i] = i;
  }

  for (let i = size - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    const temp = indices[i];
    indices[i] = indices[j];
    indices[j] = temp;
  }

  return indices;
}

function trainSample(network, input, label, learningRate) {
  const { z1, a1, z2, a2, probabilities } = forward(network, input);
  const sampleLoss = -Math.log(Math.max(probabilities[label], 1e-9));

  const delta3 = probabilities;
  delta3[label] -= 1;

  const delta2 = new Float64Array(HIDDEN_2);
  for (let neuron = 0; neuron < HIDDEN_2; neuron += 1) {
    let sum = 0;
    for (let next = 0; next < OUTPUT; next += 1) {
      sum += network.w3[next][neuron] * delta3[next];
    }
    delta2[neuron] = sum * reluDerivative(z2[neuron]);
  }

  const delta1 = new Float64Array(HIDDEN_1);
  for (let neuron = 0; neuron < HIDDEN_1; neuron += 1) {
    let sum = 0;
    for (let next = 0; next < HIDDEN_2; next += 1) {
      sum += network.w2[next][neuron] * delta2[next];
    }
    delta1[neuron] = sum * reluDerivative(z1[neuron]);
  }

  for (let output = 0; output < OUTPUT; output += 1) {
    const weights = network.w3[output];
    for (let hidden = 0; hidden < HIDDEN_2; hidden += 1) {
      weights[hidden] -= learningRate * (delta3[output] * a2[hidden] + L2_REGULARIZATION * weights[hidden]);
    }
    network.b3[output] -= learningRate * delta3[output];
  }

  for (let hidden = 0; hidden < HIDDEN_2; hidden += 1) {
    const weights = network.w2[hidden];
    for (let prev = 0; prev < HIDDEN_1; prev += 1) {
      weights[prev] -= learningRate * (delta2[hidden] * a1[prev] + L2_REGULARIZATION * weights[prev]);
    }
    network.b2[hidden] -= learningRate * delta2[hidden];
  }

  for (let hidden = 0; hidden < HIDDEN_1; hidden += 1) {
    const weights = network.w1[hidden];
    for (let pixel = 0; pixel < IMAGE_SIZE; pixel += 1) {
      weights[pixel] -= learningRate * (delta1[hidden] * input[pixel] + L2_REGULARIZATION * weights[pixel]);
    }
    network.b1[hidden] -= learningRate * delta1[hidden];
  }

  return sampleLoss;
}

function evaluate(network, images, labels) {
  let correct = 0;
  let loss = 0;
  for (let i = 0; i < images.length; i += 1) {
    const { probabilities } = forward(network, images[i]);
    loss += -Math.log(Math.max(probabilities[labels[i]], 1e-9));
    if (argmax(probabilities) === labels[i]) {
      correct += 1;
    }
  }

  return {
    accuracy: correct / images.length,
    loss: loss / images.length,
  };
}

function quantizeMatrix(matrix, decimals = 6) {
  return matrix.map((row) => Array.from(row, (value) => Number(value.toFixed(decimals))));
}

function quantizeVector(vector, decimals = 6) {
  return Array.from(vector, (value) => Number(value.toFixed(decimals)));
}

async function saveModel(network, metrics) {
  await ensureDirectory(MODEL_DIR);
  const payload = {
    architecture: {
      input: IMAGE_SIZE,
      hidden: [HIDDEN_1, HIDDEN_2],
      output: OUTPUT,
    },
    normalization: "pixel / 255",
    labels: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    metrics,
    weights: {
      w1: quantizeMatrix(network.w1),
      b1: quantizeVector(network.b1),
      w2: quantizeMatrix(network.w2),
      b2: quantizeVector(network.b2),
      w3: quantizeMatrix(network.w3),
      b3: quantizeVector(network.b3),
    },
  };

  await fs.writeFile(path.join(MODEL_DIR, "model.json"), JSON.stringify(payload));
}

async function main() {
  await ensureDirectory(DATA_DIR);
  console.log("Loading MNIST...");
  const [trainImages, trainLabels, testImages, testLabels] = await Promise.all([
    parseImages(TRAIN_IMAGES_FILE, TRAIN_LIMIT),
    parseLabels(TRAIN_LABELS_FILE, TRAIN_LIMIT),
    parseImages(TEST_IMAGES_FILE, TEST_LIMIT),
    parseLabels(TEST_LABELS_FILE, TEST_LIMIT),
  ]);

  const random = mulberry32(17);
  const network = createNetwork(random);

  console.log(`Training on ${trainImages.length} samples, evaluating on ${testImages.length} samples.`);
  for (let epoch = 0; epoch < EPOCHS; epoch += 1) {
    const shuffled = shuffleIndices(trainImages.length, random);
    let epochLoss = 0;
    const learningRate = LEARNING_RATE * Math.pow(0.95, epoch);
    for (let step = 0; step < shuffled.length; step += 1) {
      const index = shuffled[step];
      epochLoss += trainSample(network, trainImages[index], trainLabels[index], learningRate);
    }

    const metrics = evaluate(network, testImages, testLabels);
    console.log(
      `Epoch ${String(epoch + 1).padStart(2, "0")}/${EPOCHS} ` +
        `train loss ${(epochLoss / trainImages.length).toFixed(4)} ` +
        `test loss ${metrics.loss.toFixed(4)} ` +
        `test acc ${(metrics.accuracy * 100).toFixed(2)}%`,
    );
  }

  const metrics = evaluate(network, testImages, testLabels);
  await saveModel(network, {
    testAccuracy: Number(metrics.accuracy.toFixed(4)),
    testLoss: Number(metrics.loss.toFixed(4)),
    trainSamples: trainImages.length,
    testSamples: testImages.length,
    epochs: EPOCHS,
  });

  console.log("Model saved to model/model.json");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
