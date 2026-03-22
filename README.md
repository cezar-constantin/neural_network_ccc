# CCC Neural Network

Interactive simulator for handwritten digit recognition and neural network activation visualization, rebuilt from the original `Neuronal network` project with a darker portfolio-style interface.

## What It Does

- provides a drawing canvas for handwritten digits;
- normalizes the drawing to 28x28 pixels in MNIST style;
- runs inference through a trained network with architecture `784 -> 16 -> 16 -> 10`;
- visualizes neuron activation across each layer in real time;
- displays learned feature maps for both hidden layers;
- includes a contact form and GitHub Pages deployment workflow.

## Design Direction

- minimal, modern, high-end presentation;
- dark interface based on `#0B0F1A`;
- accent palette using blue `#3B82F6` and cyan `#22D3EE`;
- typography with `Space Grotesk` for headings and `Inter` for body copy.

## Model Architecture

- input: 784 neurons (`28 x 28`);
- hidden layer 1: 16 neurons for local strokes and pattern detection;
- hidden layer 2: 16 neurons for digit parts such as arcs, bars, and loops;
- output: 10 neurons for digits `0-9`.

The model in `model/model.json` was trained locally on MNIST examples using `scripts/train-mnist.mjs`.

## Local Run

The project is static, so it does not need a build step.

1. Start a local server from the project root:

```powershell
python -m http.server 4173
```

2. Open:

```text
http://localhost:4173
```

## Retraining The Model

The raw MNIST files are not committed to the repo. If you want to recreate the model:

1. download the MNIST files into the `data/` folder;
2. run:

```powershell
node scripts/train-mnist.mjs
```

The script will generate `model/model.json`.

## GitHub Pages Deployment

The repository includes `.github/workflows/deploy-pages.yml`.

1. push the code to the `main` branch of `cezar-constantin/neural_network_ccc`;
2. in GitHub, enable `Pages` with `GitHub Actions` as the source;
3. after the first successful workflow run, the app will be available at:

```text
https://cezar-constantin.github.io/neural_network_ccc/
```

## Important Files

- `index.html` - UI structure;
- `styles.css` - portfolio-inspired visual identity and layout;
- `app.js` - drawing, preprocessing, inference, and animation;
- `model/model.json` - trained model;
- `model/examples.json` - MNIST examples for the demo;
- `scripts/train-mnist.mjs` - training script.
