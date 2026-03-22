const SVG_NS = "http://www.w3.org/2000/svg";
const SOURCE_SIZE = 560;
const NORMALIZED_SIZE = 28;
const FEATURE_THUMBNAIL_SIZE = 84;
const INPUT_THRESHOLD = 12;
const INPUT_SIZE = NORMALIZED_SIZE * NORMALIZED_SIZE;
const HIDDEN_SIZE = 16;
const OUTPUT_SIZE = 10;
const MAX_INPUT_CONNECTIONS = 120;
const INPUT_EDGE_THRESHOLD = 0.055;
const INPUT_EDGE_MIN_INTENSITY = 0.025;
const MIN_NETWORK_ZOOM = 1;
const MAX_NETWORK_ZOOM = 2.4;
const NETWORK_ZOOM_STEP = 0.2;
const MOBILE_BREAKPOINT = 860;
const PHONE_BREAKPOINT = 540;
const CONTACT_FORM_ENDPOINT = "https://formsubmit.co/ajax/contact@cezar-constantin-chirila.com";

const COLORS = {
  ink: [248, 250, 252],
  muted: [148, 163, 184],
  bgDark: [11, 15, 26],
  white: [248, 250, 252],
  blue: [59, 130, 246],
  warm: [34, 211, 238],
  sun: [251, 191, 36],
  teal: [59, 130, 246],
  cool: [249, 115, 22],
  output: [34, 211, 238],
  neutral: [15, 23, 42],
};

const state = {
  model: null,
  samples: [],
  drawing: false,
  lastPoint: null,
  predictionQueued: false,
  hasUserAdjustedZoom: false,
  featureMapsLayer2: [],
  probabilityRows: [],
  featureCardsLayer1: [],
  featureCardsLayer2: [],
  network: {
    inputImage: null,
    inputBars: [],
    inputToHiddenGroup: null,
    hidden1Nodes: [],
    hidden2Nodes: [],
    outputNodes: [],
    hidden12Edges: [],
    hidden23Edges: [],
    maxAbsW1: 1,
    maxAbsW2: 1,
    maxAbsW3: 1,
  },
};

const elements = {
  drawCanvas: document.getElementById("draw-canvas"),
  normalizedCanvas: document.getElementById("normalized-canvas"),
  clearButton: document.getElementById("clear-button"),
  sampleButton: document.getElementById("sample-button"),
  predictionDigit: document.getElementById("prediction-digit"),
  predictionLabel: document.getElementById("prediction-label"),
  predictionConfidence: document.getElementById("prediction-confidence"),
  probabilityBars: document.getElementById("probability-bars"),
  modelAccuracy: document.getElementById("model-accuracy"),
  networkSvg: document.getElementById("network-svg"),
  networkVisualFrame: document.getElementById("network-visual-frame"),
  layer1Grid: document.getElementById("feature-grid-layer-1"),
  layer2Grid: document.getElementById("feature-grid-layer-2"),
  inputStatus: document.getElementById("input-status"),
  outputSummaryCard: document.getElementById("output-summary-card"),
  contactForm: document.getElementById("contact-form"),
  contactName: document.getElementById("contact-name"),
  contactCompany: document.getElementById("contact-company"),
  contactEmail: document.getElementById("contact-email"),
  contactTopic: document.getElementById("contact-topic"),
  contactQuestion: document.getElementById("contact-question"),
  contactSubmitButton: document.getElementById("contact-submit-button"),
  contactStatus: document.getElementById("contact-status"),
  zoomOutButton: document.getElementById("zoom-out-button"),
  zoomInButton: document.getElementById("zoom-in-button"),
  zoomResetButton: document.getElementById("zoom-reset-button"),
  zoomRange: document.getElementById("zoom-range"),
  zoomValue: document.getElementById("zoom-value"),
};

const drawContext = elements.drawCanvas.getContext("2d");
const normalizedContext = elements.normalizedCanvas.getContext("2d");
const scaleCanvas = document.createElement("canvas");
scaleCanvas.width = NORMALIZED_SIZE;
scaleCanvas.height = NORMALIZED_SIZE;
const scaleContext = scaleCanvas.getContext("2d");
const centeredCanvas = document.createElement("canvas");
centeredCanvas.width = NORMALIZED_SIZE;
centeredCanvas.height = NORMALIZED_SIZE;
const centeredContext = centeredCanvas.getContext("2d");
const sampleCanvas = document.createElement("canvas");
sampleCanvas.width = NORMALIZED_SIZE;
sampleCanvas.height = NORMALIZED_SIZE;
const sampleContext = sampleCanvas.getContext("2d");

function createSvgElement(tagName, attributes = {}) {
  const element = document.createElementNS(SVG_NS, tagName);
  for (const [name, value] of Object.entries(attributes)) {
    element.setAttribute(name, String(value));
  }
  return element;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function relu(value) {
  return value > 0 ? value : 0;
}

function softmax(values) {
  let max = -Infinity;
  for (let index = 0; index < values.length; index += 1) {
    if (values[index] > max) {
      max = values[index];
    }
  }

  const result = new Float32Array(values.length);
  let sum = 0;
  for (let index = 0; index < values.length; index += 1) {
    const probability = Math.exp(values[index] - max);
    result[index] = probability;
    sum += probability;
  }

  for (let index = 0; index < result.length; index += 1) {
    result[index] /= sum;
  }

  return result;
}

function argmax(values) {
  let index = 0;
  for (let cursor = 1; cursor < values.length; cursor += 1) {
    if (values[cursor] > values[index]) {
      index = cursor;
    }
  }
  return index;
}

function mixColor(start, end, factor) {
  const amount = clamp(factor, 0, 1);
  return [
    Math.round(start[0] + (end[0] - start[0]) * amount),
    Math.round(start[1] + (end[1] - start[1]) * amount),
    Math.round(start[2] + (end[2] - start[2]) * amount),
  ];
}

function toRgba(color, alpha = 1) {
  return `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${alpha})`;
}

function toRgb(color) {
  return `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
}

function getOutputAccent(index) {
  if (OUTPUT_SIZE <= 1) {
    return COLORS.output;
  }
  return mixColor(COLORS.cool, COLORS.output, index / (OUTPUT_SIZE - 1));
}

function normalizeLayer(values) {
  let max = 0;
  for (let index = 0; index < values.length; index += 1) {
    if (values[index] > max) {
      max = values[index];
    }
  }

  if (max <= 1e-9) {
    return values.map(() => 0);
  }

  return values.map((value) => Math.pow(value / max, 0.8));
}

function typedMatrix(matrix) {
  return matrix.map((row) => Float32Array.from(row));
}

function typedVector(vector) {
  return Float32Array.from(vector);
}

function prepareModel(payload) {
  return {
    ...payload,
    weights: {
      w1: typedMatrix(payload.weights.w1),
      b1: typedVector(payload.weights.b1),
      w2: typedMatrix(payload.weights.w2),
      b2: typedVector(payload.weights.b2),
      w3: typedMatrix(payload.weights.w3),
      b3: typedVector(payload.weights.b3),
    },
  };
}

function clearSourceCanvas() {
  drawContext.fillStyle = "black";
  drawContext.fillRect(0, 0, SOURCE_SIZE, SOURCE_SIZE);
}

function updateBrushProfile() {
  const compactViewport = window.innerWidth <= MOBILE_BREAKPOINT;
  const phoneViewport = window.innerWidth <= PHONE_BREAKPOINT;
  const coarsePointer = window.matchMedia("(pointer: coarse)").matches;
  drawContext.lineWidth = phoneViewport || coarsePointer ? 40 : compactViewport ? 37 : 34;
  drawContext.shadowBlur = compactViewport ? 18 : 24;
}

function configureDrawContext() {
  drawContext.lineCap = "round";
  drawContext.lineJoin = "round";
  drawContext.strokeStyle = "rgba(255, 255, 255, 0.98)";
  drawContext.fillStyle = "rgba(255, 255, 255, 0.98)";
  drawContext.shadowColor = "rgba(255, 255, 255, 0.28)";
  updateBrushProfile();
}

function setupProbabilityBars() {
  for (let digit = 0; digit < OUTPUT_SIZE; digit += 1) {
    const row = document.createElement("div");
    row.className = "probability-row";

    const label = document.createElement("span");
    label.textContent = digit;

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill = document.createElement("div");
    fill.className = "bar-fill";
    track.appendChild(fill);

    const value = document.createElement("strong");
    value.textContent = "0%";

    row.append(label, track, value);
    elements.probabilityBars.appendChild(row);
    state.probabilityRows.push({ fill, value });
  }
}

function createFeatureCard(container, label, layerClass) {
  const card = document.createElement("article");
  card.className = `feature-card ${layerClass}`;

  const header = document.createElement("div");
  header.className = "feature-card-header";

  const title = document.createElement("div");
  title.className = "feature-card-title";
  title.textContent = label;

  const value = document.createElement("div");
  value.className = "feature-card-value";
  value.textContent = "0% active";

  const canvas = document.createElement("canvas");
  canvas.width = FEATURE_THUMBNAIL_SIZE;
  canvas.height = FEATURE_THUMBNAIL_SIZE;

  header.append(title, value);
  card.append(header, canvas);
  container.appendChild(card);

  return { card, canvas, value };
}

function setupFeatureGrids() {
  for (let index = 0; index < HIDDEN_SIZE; index += 1) {
    state.featureCardsLayer1.push(
      createFeatureCard(elements.layer1Grid, `Neuron ${String(index + 1).padStart(2, "0")}`, "is-layer-1"),
    );
    state.featureCardsLayer2.push(
      createFeatureCard(elements.layer2Grid, `Neuron ${String(index + 1).padStart(2, "0")}`, "is-layer-2"),
    );
  }
}

function createLayerTitle(container, x, title, subtitle) {
  const titleText = createSvgElement("text", {
    x,
    y: 74,
    class: "network-label",
    "text-anchor": "middle",
  });
  titleText.textContent = title;

  const subtitleText = createSvgElement("text", {
    x,
    y: 98,
    class: "network-subtitle",
    "text-anchor": "middle",
  });
  subtitleText.textContent = subtitle;

  container.append(titleText, subtitleText);
}

function createLayerCard(container, x, y, width, height) {
  const card = createSvgElement("rect", {
    x,
    y,
    width,
    height,
    rx: 28,
    class: "network-layer-card",
    fill: "rgba(13, 16, 25, 0.94)",
    stroke: "rgba(255, 255, 255, 0.08)",
  });
  container.appendChild(card);
  return card;
}

function createValueNode(container, { x, y, radius, label, accent, valueX }) {
  const group = createSvgElement("g", {});
  const circle = createSvgElement("circle", {
    cx: x,
    cy: y,
    r: radius,
    class: "network-node-circle",
    fill: "rgba(30, 41, 59, 0.92)",
    stroke: toRgba(accent, 0.35),
    "stroke-width": 2,
  });

  const text = createSvgElement("text", {
    x,
    y,
    class: "network-node-text",
  });
  text.textContent = label;

  const value = createSvgElement("text", {
    x: valueX,
    y,
    class: "network-node-value",
  });
  value.textContent = "0%";

  group.append(circle, text, value);
  container.appendChild(group);
  return { circle, value, x, y, radius };
}

function createInputBar(container, { x, y, width, height }) {
  const rect = createSvgElement("rect", {
    x,
    y,
    width,
    height,
    rx: Math.max(height * 0.45, 0.12),
    class: "network-input-bar",
    fill: "#000000",
  });
  container.appendChild(rect);
  return {
    rect,
    x,
    y,
    width,
    height,
    centerY: y + height / 2,
  };
}

function buildCurve(startX, startY, endX, endY) {
  const controlOffset = (endX - startX) * 0.52;
  return `M ${startX} ${startY} C ${startX + controlOffset} ${startY}, ${endX - controlOffset} ${endY}, ${endX} ${endY}`;
}

function createConnectionPath(container, startX, startY, endX, endY, stroke, strokeWidth, opacity) {
  const path = createSvgElement("path", {
    d: buildCurve(startX, startY, endX, endY),
    class: "network-edge",
    stroke,
    "stroke-width": strokeWidth,
    opacity,
  });
  container.appendChild(path);
  return path;
}

function buildNetworkSvg() {
  const svg = elements.networkSvg;
  svg.innerHTML = "";

  const defs = createSvgElement("defs");
  const inputFilter = createSvgElement("filter", {
    id: "input-shadow",
    x: "-20%",
    y: "-20%",
    width: "140%",
    height: "140%",
  });
  inputFilter.appendChild(
    createSvgElement("feDropShadow", {
      dx: 0,
      dy: 18,
      stdDeviation: 18,
      "flood-color": "#22d3ee",
      "flood-opacity": 0.18,
    }),
  );
  defs.appendChild(inputFilter);
  svg.appendChild(defs);

  const backgroundLayer = createSvgElement("g", {});
  const inputEdgeLayer = createSvgElement("g", {});
  const hiddenEdgeLayer = createSvgElement("g", {});
  const outputEdgeLayer = createSvgElement("g", {});
  const nodeLayer = createSvgElement("g", {});
  const labelLayer = createSvgElement("g", {});
  svg.append(backgroundLayer, inputEdgeLayer, hiddenEdgeLayer, outputEdgeLayer, nodeLayer, labelLayer);

  createLayerTitle(labelLayer, 174, "Input sketch", "normalized 28x28 canvas");
  createLayerTitle(labelLayer, 394, "Input layer", "784 pixel activations");
  createLayerTitle(labelLayer, 608, "Hidden layer 1", "16 neurons, one column");
  createLayerTitle(labelLayer, 902, "Hidden layer 2", "16 neurons, one column");
  createLayerTitle(labelLayer, 1218, "Output", "10 digit probabilities");

  createLayerCard(backgroundLayer, 28, 132, 294, 354);
  const inputLayerCard = createLayerCard(backgroundLayer, 340, 132, 108, 770);
  createLayerCard(backgroundLayer, 496, 132, 224, 770);
  createLayerCard(backgroundLayer, 790, 132, 224, 770);
  createLayerCard(backgroundLayer, 1094, 214, 250, 554);
  inputLayerCard.setAttribute("fill", "rgba(13, 16, 25, 0.98)");
  inputLayerCard.setAttribute("stroke", "rgba(255, 255, 255, 0.08)");

  const inputFrame = createSvgElement("rect", {
    x: 76,
    y: 182,
    width: 198,
    height: 198,
    rx: 22,
    class: "network-input-frame",
    fill: "#0d1019",
    stroke: "rgba(255,255,255,0.14)",
    filter: "url(#input-shadow)",
  });

  const inputImage = createSvgElement("image", {
    x: 89,
    y: 195,
    width: 172,
    height: 172,
    href: "",
    preserveAspectRatio: "none",
  });
  inputImage.style.imageRendering = "pixelated";

  const inputCaption = createSvgElement("text", {
    x: 175,
    y: 420,
    class: "network-subtitle",
    "text-anchor": "middle",
  });
  inputCaption.textContent = "flattened into 784 rows";
  nodeLayer.append(inputFrame, inputImage, inputCaption);
  state.network.inputImage = inputImage;

  const inputTop = 172;
  const inputHeight = 720;
  const inputStep = inputHeight / INPUT_SIZE;
  const barWidth = 28;
  const barHeight = Math.max(inputStep * 0.72, 0.18);
  const barX = 380;
  state.network.inputBars = [];

  for (let index = 0; index < INPUT_SIZE; index += 1) {
    const barY = inputTop + index * inputStep;
    state.network.inputBars.push(
      createInputBar(nodeLayer, {
        x: barX,
        y: barY,
        width: barWidth,
        height: barHeight,
      }),
    );
  }

  const firstIndex = createSvgElement("text", {
    x: 394,
    y: 164,
    class: "network-index-label",
    "text-anchor": "middle",
  });
  firstIndex.textContent = "1";

  const lastIndex = createSvgElement("text", {
    x: 394,
    y: 915,
    class: "network-index-label",
    "text-anchor": "middle",
  });
  lastIndex.textContent = "784";
  labelLayer.append(firstIndex, lastIndex);

  const hiddenTop = 184;
  const hiddenGap = 43;
  const hiddenRadius = 16;
  state.network.hidden1Nodes = Array.from({ length: HIDDEN_SIZE }, (_, index) =>
    createValueNode(nodeLayer, {
      x: 586,
      y: hiddenTop + index * hiddenGap,
      radius: hiddenRadius,
      label: index + 1,
      accent: COLORS.warm,
      valueX: 626,
    }),
  );
  state.network.hidden2Nodes = Array.from({ length: HIDDEN_SIZE }, (_, index) =>
    createValueNode(nodeLayer, {
      x: 880,
      y: hiddenTop + index * hiddenGap,
      radius: hiddenRadius,
      label: index + 1,
      accent: COLORS.teal,
      valueX: 920,
    }),
  );

  const outputTop = 282;
  const outputGap = 48;
  const outputRadius = 16;
  state.network.outputNodes = Array.from({ length: OUTPUT_SIZE }, (_, index) =>
    createValueNode(nodeLayer, {
      x: 1182,
      y: outputTop + index * outputGap,
      radius: outputRadius,
      label: index,
      accent: getOutputAccent(index),
      valueX: 1224,
    }),
  );

  state.network.inputToHiddenGroup = inputEdgeLayer;
  state.network.hidden12Edges = [];
  state.network.hidden23Edges = [];

  for (let target = 0; target < HIDDEN_SIZE; target += 1) {
    for (let source = 0; source < HIDDEN_SIZE; source += 1) {
      const start = state.network.hidden1Nodes[source];
      const end = state.network.hidden2Nodes[target];
      const path = createConnectionPath(
        hiddenEdgeLayer,
        start.x + start.radius,
        start.y,
        end.x - end.radius,
        end.y,
        "rgba(148, 163, 184, 0.08)",
        1,
        0.22,
      );
      state.network.hidden12Edges.push({
        path,
        source,
        target,
        weight: state.model.weights.w2[target][source],
      });
    }
  }

  for (let target = 0; target < OUTPUT_SIZE; target += 1) {
    for (let source = 0; source < HIDDEN_SIZE; source += 1) {
      const start = state.network.hidden2Nodes[source];
      const end = state.network.outputNodes[target];
      const path = createConnectionPath(
        outputEdgeLayer,
        start.x + start.radius,
        start.y,
        end.x - end.radius,
        end.y,
        "rgba(148, 163, 184, 0.08)",
        1,
        0.22,
      );
      state.network.hidden23Edges.push({
        path,
        source,
        target,
        weight: state.model.weights.w3[target][source],
      });
    }
  }

  state.network.maxAbsW1 = Math.max(...state.model.weights.w1.flatMap((row) => Array.from(row, Math.abs)), 0.0001);
  state.network.maxAbsW2 = Math.max(...state.network.hidden12Edges.map((edge) => Math.abs(edge.weight)), 0.0001);
  state.network.maxAbsW3 = Math.max(...state.network.hidden23Edges.map((edge) => Math.abs(edge.weight)), 0.0001);
}

function drawHeatmap(canvas, values, accent) {
  const context = canvas.getContext("2d");
  const pixelCanvas = document.createElement("canvas");
  pixelCanvas.width = NORMALIZED_SIZE;
  pixelCanvas.height = NORMALIZED_SIZE;
  const pixelContext = pixelCanvas.getContext("2d");
  const imageData = pixelContext.createImageData(NORMALIZED_SIZE, NORMALIZED_SIZE);

  let maxAbs = 0;
  for (let index = 0; index < values.length; index += 1) {
    const absValue = Math.abs(values[index]);
    if (absValue > maxAbs) {
      maxAbs = absValue;
    }
  }
  maxAbs = Math.max(maxAbs, 1e-6);

  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    const amount = Math.pow(Math.abs(value) / maxAbs, 0.72);
    const color = mixColor(COLORS.bgDark, value >= 0 ? accent : COLORS.cool, amount);
    const offset = index * 4;
    imageData.data[offset] = color[0];
    imageData.data[offset + 1] = color[1];
    imageData.data[offset + 2] = color[2];
    imageData.data[offset + 3] = 255;
  }

  pixelContext.putImageData(imageData, 0, 0);
  context.imageSmoothingEnabled = false;
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(pixelCanvas, 0, 0, canvas.width, canvas.height);
}

function projectLayerTwoFeatures(model) {
  return model.weights.w2.map((row) => {
    const composite = new Float32Array(INPUT_SIZE);
    for (let source = 0; source < row.length; source += 1) {
      const weight = row[source];
      const sourceWeights = model.weights.w1[source];
      for (let pixel = 0; pixel < composite.length; pixel += 1) {
        composite[pixel] += sourceWeights[pixel] * weight;
      }
    }
    return composite;
  });
}

function renderFeatureMaps() {
  state.model.weights.w1.forEach((weights, index) => {
    drawHeatmap(state.featureCardsLayer1[index].canvas, weights, COLORS.warm);
  });
  state.featureMapsLayer2.forEach((weights, index) => {
    drawHeatmap(state.featureCardsLayer2[index].canvas, weights, COLORS.teal);
  });
}

function setFeatureActivity(cards, activations, accent) {
  const normalized = normalizeLayer(activations);
  cards.forEach((card, index) => {
    const intensity = normalized[index];
    card.card.style.setProperty("--activation", intensity.toFixed(3));
    card.card.style.borderColor = toRgba(accent, 0.1 + intensity * 0.4);
    card.value.textContent = `${Math.round(intensity * 100)}% active`;
  });
}

function setNodeVisual(node, intensity, accent, valueText) {
  const fill = mixColor(COLORS.neutral, accent, clamp(0.14 + intensity * 0.86, 0, 1));
  node.circle.setAttribute("fill", toRgb(fill));
  node.circle.setAttribute("stroke", toRgba(accent, 0.22 + intensity * 0.52));
  node.circle.setAttribute("r", (node.radius * (1 + intensity * 0.12)).toFixed(2));
  node.value.textContent = valueText;
}

function setInputBarVisual(bar, intensity) {
  const fill = mixColor(COLORS.bgDark, COLORS.white, clamp(intensity, 0, 1));
  bar.rect.setAttribute("fill", toRgb(fill));
  bar.rect.setAttribute("opacity", "1");
}

function getActiveInputRows(inputValues) {
  const active = [];
  for (let index = 0; index < inputValues.length; index += 1) {
    const value = inputValues[index];
    if (value > INPUT_EDGE_THRESHOLD) {
      active.push({ index, value });
    }
  }
  active.sort((left, right) => right.value - left.value);
  return active.slice(0, MAX_INPUT_CONNECTIONS);
}

function renderInputToHiddenEdges(inputValues, hidden1Norm) {
  const group = state.network.inputToHiddenGroup;
  group.replaceChildren();

  const activeInputs = getActiveInputRows(inputValues);
  const fragment = document.createDocumentFragment();

  for (const activeInput of activeInputs) {
    const start = state.network.inputBars[activeInput.index];
    for (let target = 0; target < HIDDEN_SIZE; target += 1) {
      const weight = state.model.weights.w1[target][activeInput.index];
      const strength = (Math.abs(weight) / state.network.maxAbsW1) * activeInput.value * hidden1Norm[target];
      if (strength < INPUT_EDGE_MIN_INTENSITY) {
        continue;
      }

      const end = state.network.hidden1Nodes[target];
      const color = weight >= 0 ? COLORS.blue : COLORS.cool;
      const path = createSvgElement("path", {
        d: buildCurve(start.x + start.width, start.centerY, end.x - end.radius, end.y),
        class: "network-edge",
        stroke: toRgba(color, 0.05 + strength * 0.88),
        "stroke-width": (0.35 + strength * 1.7).toFixed(2),
        opacity: (0.14 + strength * 0.86).toFixed(3),
      });
      fragment.appendChild(path);
    }
  }

  group.appendChild(fragment);
}

function updateWeightedEdges(edges, sourceNorm, targetNorm, maxWeight, positiveColor) {
  edges.forEach((edge) => {
    const strength = Math.abs(edge.weight) / maxWeight;
    const intensity = sourceNorm[edge.source] * targetNorm[edge.target] * Math.pow(strength, 0.75);
    const color = edge.weight >= 0 ? positiveColor : COLORS.cool;
    edge.path.setAttribute("stroke", toRgba(color, 0.04 + intensity * 0.9));
    edge.path.setAttribute("stroke-width", (0.7 + intensity * 2.25).toFixed(2));
    edge.path.setAttribute("opacity", (0.1 + intensity * 0.9).toFixed(3));
  });
}

function setNetworkState(result) {
  if (!result) {
    state.network.inputBars.forEach((bar) => setInputBarVisual(bar, 0));
    state.network.hidden1Nodes.forEach((node) => setNodeVisual(node, 0, COLORS.warm, "0%"));
    state.network.hidden2Nodes.forEach((node) => setNodeVisual(node, 0, COLORS.teal, "0%"));
    state.network.outputNodes.forEach((node, index) => setNodeVisual(node, 0, getOutputAccent(index), "0%"));
    renderInputToHiddenEdges(new Array(INPUT_SIZE).fill(0), new Array(HIDDEN_SIZE).fill(0));
    updateWeightedEdges(
      state.network.hidden12Edges,
      new Array(HIDDEN_SIZE).fill(0),
      new Array(HIDDEN_SIZE).fill(0),
      state.network.maxAbsW2,
      COLORS.blue,
    );
    updateWeightedEdges(
      state.network.hidden23Edges,
      new Array(HIDDEN_SIZE).fill(0),
      new Array(OUTPUT_SIZE).fill(0),
      state.network.maxAbsW3,
      COLORS.blue,
    );
    if (state.network.inputImage) {
      state.network.inputImage.setAttribute("href", "");
    }
    return;
  }

  const inputNorm = Array.from(result.input);
  const hidden1Norm = normalizeLayer(result.a1);
  const hidden2Norm = normalizeLayer(result.a2);
  const probabilities = Array.from(result.probabilities);

  state.network.inputBars.forEach((bar, index) => setInputBarVisual(bar, inputNorm[index]));
  state.network.hidden1Nodes.forEach((node, index) =>
    setNodeVisual(node, hidden1Norm[index], COLORS.warm, `${Math.round(hidden1Norm[index] * 100)}%`),
  );
  state.network.hidden2Nodes.forEach((node, index) =>
    setNodeVisual(node, hidden2Norm[index], COLORS.teal, `${Math.round(hidden2Norm[index] * 100)}%`),
  );
  state.network.outputNodes.forEach((node, index) =>
    setNodeVisual(node, probabilities[index], getOutputAccent(index), `${Math.round(probabilities[index] * 100)}%`),
  );

  renderInputToHiddenEdges(inputNorm, hidden1Norm);
  updateWeightedEdges(state.network.hidden12Edges, hidden1Norm, hidden2Norm, state.network.maxAbsW2, COLORS.blue);
  updateWeightedEdges(state.network.hidden23Edges, hidden2Norm, probabilities, state.network.maxAbsW3, COLORS.blue);
  state.network.inputImage.setAttribute("href", centeredCanvas.toDataURL("image/png"));
}

function forwardPass(model, input) {
  const { w1, b1, w2, b2, w3, b3 } = model.weights;

  const z1 = new Float32Array(HIDDEN_SIZE);
  const a1 = new Float32Array(HIDDEN_SIZE);
  for (let neuron = 0; neuron < HIDDEN_SIZE; neuron += 1) {
    let sum = b1[neuron];
    const weights = w1[neuron];
    for (let pixel = 0; pixel < input.length; pixel += 1) {
      sum += weights[pixel] * input[pixel];
    }
    z1[neuron] = sum;
    a1[neuron] = relu(sum);
  }

  const z2 = new Float32Array(HIDDEN_SIZE);
  const a2 = new Float32Array(HIDDEN_SIZE);
  for (let neuron = 0; neuron < HIDDEN_SIZE; neuron += 1) {
    let sum = b2[neuron];
    const weights = w2[neuron];
    for (let source = 0; source < a1.length; source += 1) {
      sum += weights[source] * a1[source];
    }
    z2[neuron] = sum;
    a2[neuron] = relu(sum);
  }

  const z3 = new Float32Array(OUTPUT_SIZE);
  for (let neuron = 0; neuron < OUTPUT_SIZE; neuron += 1) {
    let sum = b3[neuron];
    const weights = w3[neuron];
    for (let source = 0; source < a2.length; source += 1) {
      sum += weights[source] * a2[source];
    }
    z3[neuron] = sum;
  }

  const probabilities = softmax(z3);
  return { input, z1, a1, z2, a2, z3, probabilities };
}

function getPointerPosition(event) {
  const rect = elements.drawCanvas.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * SOURCE_SIZE;
  const y = ((event.clientY - rect.top) / rect.height) * SOURCE_SIZE;
  return {
    x: clamp(x, 0, SOURCE_SIZE),
    y: clamp(y, 0, SOURCE_SIZE),
  };
}

function drawDot(point) {
  drawContext.beginPath();
  drawContext.arc(point.x, point.y, drawContext.lineWidth * 0.3, 0, Math.PI * 2);
  drawContext.fill();
}

function drawSegment(from, to) {
  drawContext.beginPath();
  drawContext.moveTo(from.x, from.y);
  drawContext.lineTo(to.x, to.y);
  drawContext.stroke();
}

function queuePrediction() {
  if (state.predictionQueued) {
    return;
  }

  state.predictionQueued = true;
  requestAnimationFrame(() => {
    state.predictionQueued = false;
    runPrediction();
  });
}

function getBoundsFromImage(imageData) {
  const { data, width, height } = imageData;
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const offset = (y * width + x) * 4;
      const value = data[offset];
      if (value > INPUT_THRESHOLD) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (maxX === -1) {
    return null;
  }

  return {
    minX,
    minY,
    maxX,
    maxY,
    width: maxX - minX + 1,
    height: maxY - minY + 1,
  };
}

function centerNormalizedDigit() {
  const imageData = scaleContext.getImageData(0, 0, NORMALIZED_SIZE, NORMALIZED_SIZE);
  const { data } = imageData;
  let mass = 0;
  let centerX = 0;
  let centerY = 0;

  for (let y = 0; y < NORMALIZED_SIZE; y += 1) {
    for (let x = 0; x < NORMALIZED_SIZE; x += 1) {
      const offset = (y * NORMALIZED_SIZE + x) * 4;
      const intensity = data[offset] / 255;
      mass += intensity;
      centerX += x * intensity;
      centerY += y * intensity;
    }
  }

  centeredContext.fillStyle = "black";
  centeredContext.fillRect(0, 0, NORMALIZED_SIZE, NORMALIZED_SIZE);

  if (mass <= 1e-9) {
    return new Float32Array(INPUT_SIZE);
  }

  const shiftX = Math.round(13.5 - centerX / mass);
  const shiftY = Math.round(13.5 - centerY / mass);
  centeredContext.drawImage(scaleCanvas, shiftX, shiftY);

  const centeredImage = centeredContext.getImageData(0, 0, NORMALIZED_SIZE, NORMALIZED_SIZE);
  const input = new Float32Array(INPUT_SIZE);

  for (let index = 0; index < input.length; index += 1) {
    const intensity = centeredImage.data[index * 4] / 255;
    input[index] = Math.pow(intensity, 0.9);
  }

  normalizedContext.imageSmoothingEnabled = false;
  normalizedContext.clearRect(0, 0, NORMALIZED_SIZE, NORMALIZED_SIZE);
  normalizedContext.drawImage(centeredCanvas, 0, 0);

  return input;
}

function preprocessDrawing() {
  const sourceImage = drawContext.getImageData(0, 0, SOURCE_SIZE, SOURCE_SIZE);
  const bounds = getBoundsFromImage(sourceImage);
  if (!bounds) {
    scaleContext.fillStyle = "black";
    scaleContext.fillRect(0, 0, NORMALIZED_SIZE, NORMALIZED_SIZE);
    centeredContext.fillStyle = "black";
    centeredContext.fillRect(0, 0, NORMALIZED_SIZE, NORMALIZED_SIZE);
    normalizedContext.fillStyle = "black";
    normalizedContext.fillRect(0, 0, NORMALIZED_SIZE, NORMALIZED_SIZE);
    return { input: new Float32Array(INPUT_SIZE), empty: true };
  }

  const targetScale = 20 / Math.max(bounds.width, bounds.height);
  const targetWidth = Math.max(1, Math.round(bounds.width * targetScale));
  const targetHeight = Math.max(1, Math.round(bounds.height * targetScale));

  scaleContext.fillStyle = "black";
  scaleContext.fillRect(0, 0, NORMALIZED_SIZE, NORMALIZED_SIZE);
  scaleContext.imageSmoothingEnabled = true;
  scaleContext.drawImage(
    elements.drawCanvas,
    bounds.minX,
    bounds.minY,
    bounds.width,
    bounds.height,
    (NORMALIZED_SIZE - targetWidth) / 2,
    (NORMALIZED_SIZE - targetHeight) / 2,
    targetWidth,
    targetHeight,
  );

  return {
    input: centerNormalizedDigit(),
    empty: false,
  };
}

function setProbabilityBars(probabilities) {
  probabilities.forEach((probability, index) => {
    const percent = Math.round(probability * 100);
    state.probabilityRows[index].fill.style.width = `${percent}%`;
    state.probabilityRows[index].value.textContent = `${percent}%`;
  });
}

function setNetworkZoom(nextZoom) {
  const zoom = clamp(nextZoom, MIN_NETWORK_ZOOM, MAX_NETWORK_ZOOM);
  const percentage = Math.round(zoom * 100);

  elements.networkVisualFrame.style.width = `${percentage}%`;
  elements.zoomRange.value = String(percentage);
  elements.zoomValue.textContent = `${percentage}%`;
  elements.zoomOutButton.disabled = zoom <= MIN_NETWORK_ZOOM + 1e-9;
  elements.zoomInButton.disabled = zoom >= MAX_NETWORK_ZOOM - 1e-9;
  elements.zoomResetButton.disabled = Math.abs(zoom - 1) < 1e-9;
}

function getDefaultNetworkZoom() {
  if (window.innerWidth <= PHONE_BREAKPOINT) {
    return 1.8;
  }
  if (window.innerWidth <= MOBILE_BREAKPOINT) {
    return 1.35;
  }
  return 1;
}

function syncResponsiveUi(forceZoom = false) {
  updateBrushProfile();
  if (forceZoom || !state.hasUserAdjustedZoom) {
    setNetworkZoom(getDefaultNetworkZoom());
  }
}

function setPredictionEmpty() {
  elements.outputSummaryCard.classList.add("empty-state");
  elements.predictionDigit.textContent = "?";
  elements.predictionLabel.textContent = "Waiting for your drawing";
  elements.predictionConfidence.textContent = "Confidence: 0%";
  normalizedContext.fillStyle = "black";
  normalizedContext.fillRect(0, 0, NORMALIZED_SIZE, NORMALIZED_SIZE);
  setProbabilityBars(new Array(OUTPUT_SIZE).fill(0));
  setNetworkState(null);
  setFeatureActivity(state.featureCardsLayer1, new Array(HIDDEN_SIZE).fill(0), COLORS.warm);
  setFeatureActivity(state.featureCardsLayer2, new Array(HIDDEN_SIZE).fill(0), COLORS.teal);
  elements.inputStatus.textContent = "Canvas is empty";
}

function setPredictionResult(result) {
  const predictedDigit = argmax(result.probabilities);
  const confidence = result.probabilities[predictedDigit];
  elements.outputSummaryCard.classList.remove("empty-state");
  elements.predictionDigit.textContent = predictedDigit;
  elements.predictionLabel.textContent = `The model most likely sees digit ${predictedDigit}`;
  elements.predictionConfidence.textContent = `Confidence: ${Math.round(confidence * 100)}%`;
  setProbabilityBars(Array.from(result.probabilities));
  setNetworkState(result);
  setFeatureActivity(state.featureCardsLayer1, Array.from(result.a1), COLORS.warm);
  setFeatureActivity(state.featureCardsLayer2, Array.from(result.a2), COLORS.teal);
  elements.inputStatus.textContent = "Live prediction";
}

function runPrediction() {
  if (!state.model) {
    return;
  }

  const processed = preprocessDrawing();
  if (processed.empty) {
    setPredictionEmpty();
    return;
  }

  const result = forwardPass(state.model, processed.input);
  setPredictionResult(result);
}

function startDrawing(event) {
  event.preventDefault();
  elements.drawCanvas.setPointerCapture(event.pointerId);
  state.drawing = true;
  const point = getPointerPosition(event);
  state.lastPoint = point;
  drawDot(point);
  queuePrediction();
}

function continueDrawing(event) {
  if (!state.drawing) {
    return;
  }
  event.preventDefault();
  const point = getPointerPosition(event);
  drawSegment(state.lastPoint, point);
  state.lastPoint = point;
  queuePrediction();
}

function stopDrawing(event) {
  if (!state.drawing) {
    return;
  }
  state.drawing = false;
  state.lastPoint = null;
  if (event && elements.drawCanvas.hasPointerCapture(event.pointerId)) {
    elements.drawCanvas.releasePointerCapture(event.pointerId);
  }
  queuePrediction();
}

function clearDrawing() {
  clearSourceCanvas();
  setPredictionEmpty();
}

function renderSample(sample) {
  sampleContext.fillStyle = "black";
  sampleContext.fillRect(0, 0, NORMALIZED_SIZE, NORMALIZED_SIZE);
  const sampleImage = sampleContext.createImageData(NORMALIZED_SIZE, NORMALIZED_SIZE);
  for (let index = 0; index < sample.pixels.length; index += 1) {
    const value = Math.round(sample.pixels[index] * 255);
    sampleImage.data[index * 4] = value;
    sampleImage.data[index * 4 + 1] = value;
    sampleImage.data[index * 4 + 2] = value;
    sampleImage.data[index * 4 + 3] = 255;
  }
  sampleContext.putImageData(sampleImage, 0, 0);

  clearSourceCanvas();
  drawContext.imageSmoothingEnabled = true;
  drawContext.drawImage(sampleCanvas, 84, 84, 392, 392);
  queuePrediction();
}

function useRandomSample() {
  if (!state.samples.length) {
    return;
  }
  const sample = state.samples[Math.floor(Math.random() * state.samples.length)];
  renderSample(sample);
}

async function loadSamples() {
  try {
    const response = await fetch("./model/examples.json");
    if (!response.ok) {
      throw new Error("Examples unavailable");
    }
    state.samples = await response.json();
    elements.sampleButton.disabled = false;
  } catch (error) {
    elements.sampleButton.disabled = true;
    elements.sampleButton.textContent = "Samples unavailable";
  }
}

function setupNetworkZoomControls() {
  elements.zoomOutButton.addEventListener("click", () => {
    state.hasUserAdjustedZoom = true;
    const nextZoom = Number(elements.zoomRange.value) / 100 - NETWORK_ZOOM_STEP;
    setNetworkZoom(nextZoom);
  });

  elements.zoomInButton.addEventListener("click", () => {
    state.hasUserAdjustedZoom = true;
    const nextZoom = Number(elements.zoomRange.value) / 100 + NETWORK_ZOOM_STEP;
    setNetworkZoom(nextZoom);
  });

  elements.zoomResetButton.addEventListener("click", () => {
    state.hasUserAdjustedZoom = false;
    setNetworkZoom(getDefaultNetworkZoom());
  });

  elements.zoomRange.addEventListener("input", () => {
    state.hasUserAdjustedZoom = true;
    setNetworkZoom(Number(elements.zoomRange.value) / 100);
  });

  syncResponsiveUi(true);
}

function setupResponsiveLayout() {
  let scheduled = false;

  const sync = () => {
    scheduled = false;
    syncResponsiveUi();
  };

  const scheduleSync = () => {
    if (scheduled) {
      return;
    }
    scheduled = true;
    requestAnimationFrame(sync);
  };

  window.addEventListener("resize", scheduleSync, { passive: true });
}

function setContactStatus(message, tone = "") {
  elements.contactStatus.textContent = message;
  elements.contactStatus.classList.remove("is-success", "is-error");
  if (tone) {
    elements.contactStatus.classList.add(`is-${tone}`);
  }
}

async function submitContactForm(event) {
  event.preventDefault();

  if (!elements.contactForm.reportValidity()) {
    return;
  }

  const formData = new FormData(elements.contactForm);
  const fullName = elements.contactName.value.trim();
  const company = elements.contactCompany.value.trim();
  const email = elements.contactEmail.value.trim();
  const topic = elements.contactTopic.value.trim();
  const question = elements.contactQuestion.value.trim();

  formData.set("Full name", fullName);
  formData.set("Company", company || "Not provided");
  formData.set("Email address", email);
  formData.set("Question topic", topic || "General question");
  formData.set("Question", question);
  formData.set("_subject", `CCC Neural Network question from ${fullName}`);
  formData.set("_replyto", email);
  formData.set("_template", "table");
  formData.set("_captcha", "false");
  formData.set("_url", window.location.href);

  elements.contactSubmitButton.disabled = true;
  setContactStatus("Sending question...");

  try {
    const response = await fetch(CONTACT_FORM_ENDPOINT, {
      method: "POST",
      headers: {
        Accept: "application/json",
      },
      body: formData,
    });

    const payload = await response.json().catch(() => null);

    if (!response.ok || payload?.success === "false" || payload?.success === false) {
      throw new Error("Contact form submission failed");
    }

    setContactStatus("Question sent successfully. You can expect a reply at the address you entered.", "success");
    elements.contactForm.reset();
  } catch (error) {
    console.error(error);
    setContactStatus(
      "The form could not be sent right now. Please try again in a moment or email contact@cezar-constantin-chirila.com directly.",
      "error",
    );
  } finally {
    elements.contactSubmitButton.disabled = false;
  }
}

function setupContactForm() {
  elements.contactForm.addEventListener("submit", submitContactForm);
}

function attachCanvasEvents() {
  elements.drawCanvas.addEventListener("pointerdown", startDrawing);
  elements.drawCanvas.addEventListener("pointermove", continueDrawing);
  elements.drawCanvas.addEventListener("pointerup", stopDrawing);
  elements.drawCanvas.addEventListener("pointerleave", stopDrawing);
  elements.drawCanvas.addEventListener("pointercancel", stopDrawing);
  elements.clearButton.addEventListener("click", clearDrawing);
  elements.sampleButton.addEventListener("click", useRandomSample);
}

async function loadModel() {
  const response = await fetch("./model/model.json");
  if (!response.ok) {
    throw new Error("Could not load the model.");
  }
  const payload = await response.json();
  state.model = prepareModel(payload);
  state.featureMapsLayer2 = projectLayerTwoFeatures(state.model);
  elements.modelAccuracy.textContent = "89% on 2000 test samples";
}

async function initialize() {
  setupProbabilityBars();
  setupFeatureGrids();
  setupNetworkZoomControls();
  setupResponsiveLayout();
  setupContactForm();
  configureDrawContext();
  clearSourceCanvas();

  try {
    await loadModel();
    buildNetworkSvg();
    renderFeatureMaps();
    await loadSamples();
    attachCanvasEvents();
    setPredictionEmpty();
  } catch (error) {
    elements.modelAccuracy.textContent = "Model failed to load";
    elements.inputStatus.textContent = "Initialization error";
    elements.predictionLabel.textContent = "Model is missing";
    elements.predictionConfidence.textContent = "Check the files inside the model/ folder";
    console.error(error);
  }
}

initialize();
