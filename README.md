# Smart Waste Classification Using Convolutional Neural Networks

## 1. Problem statement
Sorting waste by hand is slow and error-prone. We build an image classifier that looks at a photo of one waste item and predicts its material class.

## 2. Objective
Train a simple CNN in PyTorch on a public waste dataset, evaluate it honestly on unseen test images, compare it with two simpler methods (HOG+SVM, MLP), and measure with an ablation study whether augmentation and dropout improve generalisation.

## 3. Dataset
**TrashNet** (Yang & Thung, Stanford), `dataset-resized` version, from https://github.com/garythung/trashnet. Photos of single waste items on a plain background.
**Class choice (important):** TrashNet has 6 folders (cardboard, glass, metal, paper, plastic, trash) and **no organic class**. We therefore use 5 material classes: **cardboard, glass, metal, paper, plastic**; the small "trash" folder is excluded. (All 6 can be used with `INCLUDE_TRASH=1`.) Say this openly in the viva; do not claim an organic class.

## 4. Dataset statistics (computed by `dataset_analysis.py` from the real files; saved in `results/dataset_stats.json`)
- Raw images in the 5 folders: 2390; 3 byte-identical duplicates removed before splitting -> **2387 images**
- Classes: **5** | per class: cardboard 403, glass 501, metal 409, paper 594, plastic 480
- Original size: 512x384 for all 2387 images; resized to 128x128x3 for the model
- Split (stratified, seed 42): **train 1670 / validation 358 / test 359**
- Imbalance: largest class (paper) / smallest (cardboard) = 1.47 (mild imbalance)
- Leakage check: 0 files and 0 identical images shared between splits. (The check originally found 2 identical-image groups crossing splits, which is why duplicates are removed first.)
- Plots: `results/plots/class_distribution.png`, `split_distribution.png`, `sample_images.png`

Why suitable for CNN: real photos with texture/shape/colour cues; balanced enough; small enough to train on a laptop.
Limitations: only ~2.4k images; plain backgrounds (real bins are messier); items photographed several times in different poses, so near-duplicates may sit in different splits and inflate scores slightly (we only remove *exact* duplicates); no organic class; cardboard is the smallest class; squashing 4:3 images to square changes shape slightly.

## 5. Preprocessing (implemented in `preprocessing.py`)
1. RGB load (CNN needs 3 channels) 2. Resize 128x128 (fixed input size) 3. ToTensor (0-255 -> 0-1) 4. Normalise (x-0.5)/0.5 -> [-1,1] (stable, faster training; fixed constants, nothing learnt from test data).
**Augmentation, training set only:** horizontal flip (p=0.5), rotation up to 15 degrees, brightness/contrast jitter 0.2. Validation and test are never augmented, so their scores reflect real performance.
**No leakage:** the split is made on file paths *before* any transform; test images are used only in `evaluate.py` / `compare_models.py`; choices (early stopping, SVM's C) use validation only.

## 6. CNN architecture (`model.py`, diagram in `results/plots/cnn_architecture.png`)
Input 3x128x128 -> Conv(32,3x3)+ReLU -> MaxPool -> Conv(64)+ReLU -> MaxPool -> Conv(128)+ReLU -> MaxPool (128x16x16) -> Flatten (32768) -> Dense(128)+ReLU -> Dropout(0.5) -> Dense(5) -> Softmax.
Trainable parameters: printed by `python model.py` (hand calculation: 896 + 18,496 + 73,856 + 4,194,432 + 645 = 4,288,325 - confirm against the printout). Most parameters sit in the first Dense layer.
Softmax is applied when producing probabilities; during training `CrossEntropyLoss` includes it.

## 7. Training configuration
Adam, lr 0.001, batch 32, CrossEntropyLoss, max 20 epochs, early stopping on validation loss (patience 5, best-epoch weights restored), seed 42. Note: training accuracy is measured while augmentation and dropout are active, so it can be *lower* than validation accuracy; that is normal.

## 8. Results, confusion matrix, model comparison, ablation
Run the project, then copy from **`results/RESULTS.md`** (auto-generated; contains every real number) and the files below. Nothing here is typed in by hand.
- Training/validation accuracy and loss curves: `results/plots/<experiment>_{train_accuracy,val_accuracy,train_loss,val_loss}.png`
- Confusion matrix: `results/plots/confusion_matrix.png` | report: `results/classification_report.txt` | metrics: `results/metrics.json`
- Example predictions: `results/plots/example_predictions.png`, `misclassified_examples.png`; all predictions: `results/predictions.csv`
- Model comparison: `results/model_comparison.csv`, `plots/model_comparison.png`
- Ablation: `results/ablation_results.csv`
Metrics are test-set accuracy and **macro** precision, recall and F1 (average over the 5 classes).
If an experiment is worse than another, report it as it is. A result such as "dropout did not help here" is a valid finding. With ~359 test images, one image is about 0.28 percentage points, and runs use one seed, so differences under ~1 point are not conclusive.

## 9. Ablation study (CO5 contribution)
Project-level contribution (not a new algorithm): **"Regularised and augmented CNN training with quantitative ablation analysis."** Exp 1 baseline (no augmentation, no dropout); Exp 2 + augmentation; Exp 3 + augmentation + dropout. Same seed, split and hyper-parameters; only the stated element changes. Exp 1 shows how much the plain CNN overfits (train/validation gap); Exp 2 shows the effect of more varied training data; Exp 3 adds regularisation inside the network.

## 10. Comparison with published work (CO4)
Our numbers come only from our runs on our split. For literature, cite the original TrashNet report (Yang & Thung, "Classification of Trash for Recyclability Status", Stanford CS229, 2016) and other papers you can open and verify yourself; copy their numbers exactly with citation. Write clearly: *"Published results use a different split/preprocessing/classes (6 classes, different image size), so they are not directly comparable to ours."* Do not paste any number you have not read in the paper.
Why CNN may differ from SVM/MLP: HOG uses hand-designed gradient features on grayscale (colour is lost); an MLP on flattened pixels ignores pixel neighbourhoods and needs a separate weight per pixel (more parameters, easy to overfit); a CNN learns its own filters, reuses them across the image (weight sharing) and gets some shift-invariance from pooling. With small data a CNN is not guaranteed to win; report what you measure.

## 11. CO mapping
- **CO1** Dataset analysis, visualisation, preprocessing: `dataset_analysis.py`, `preprocessing.py`, dataset plots, section 4-5.
- **CO2** CNN architecture, convolution, pooling, activation, loss, optimiser: `model.py`, section 6, theory below.
- **CO3** Implementation, training, backpropagation, evaluation, prediction: `train.py` (`loss.backward()`, `optimizer.step()`), `evaluate.py`, `predict.py`.
- **CO4** HOG+SVM vs MLP vs CNN on identical split with measured metrics (+ literature with the caveat above): `hog_svm.py`, `compare_models.py`.
- **CO5** Augmentation + dropout strategy with ablation evidence: `train.py` EXPERIMENTS, `results/ablation_results.csv`.

## 12. CO2 theory in plain words
- **Convolution:** slide a small 3x3 window over the image; at each position multiply-and-add the pixels with the window's weights -> one number.
- **Filter/kernel:** that 3x3(xchannels) weight block. Learnt during training; early filters find edges, later ones textures/shapes.
- **Feature map:** the output grid produced by one filter (Conv1 makes 32 maps).
- **ReLU:** max(0,x); adds non-linearity so stacked layers can learn complex patterns; cheap, avoids vanishing gradients.
- **Max pooling (2x2):** keep the largest value per 2x2 block: halves height/width, cuts computation, adds small shift-tolerance.
- **Flatten:** turn the 128x16x16 maps into one vector of 32768 for the dense layers.
- **Dense layer:** every input connects to every output; combines features into decisions.
- **Dropout:** randomly switches off 50% of the units during training so the net cannot rely on single units -> less overfitting. Off at test time.
- **Softmax:** turns 5 raw scores into probabilities that sum to 1.
- **Loss:** CrossEntropyLoss = -log(probability given to the correct class); small when confident and right. **Optimiser:** Adam, a gradient-descent variant with per-weight adaptive step sizes.
- **Forward:** image -> CNN -> prediction -> loss. **Backprop:** `loss.backward()` computes the gradient of the loss w.r.t. every weight (chain rule). **Update:** `optimizer.step()` changes each weight a little against its gradient.

## 13. Future work
More data or a different dataset with an organic class; transfer learning (e.g. pretrained ResNet18/MobileNet); several seeds with mean +- std; split by object/photo session to avoid near-duplicate leakage; Grad-CAM to see what the model looks at; testing on cluttered real-world photos.

## 14. How to run
```bash
cd waste_cnn
pip install -r requirements.txt
python download_data.py        # TrashNet -> data/dataset-resized
python dataset_analysis.py     # stats, split, plots  (CO1)
python model.py                # param count + architecture diagram (CO2)
python train.py                # 3 ablation experiments (CO3, CO5)  [--exp baseline for one]
python evaluate.py             # test metrics, confusion matrix, report, predictions
python compare_models.py       # HOG+SVM vs MLP vs CNN (CO4)
python report.py               # results/RESULTS.md
python predict.py some_image.jpg
```
Or `bash run_all.sh`. Fast test first: `python train.py --exp augmentation_dropout --epochs 2`.

## Project structure
`config.py` settings | `dataset_analysis.py` stats+split | `preprocessing.py` transforms+loaders | `model.py` CNN/MLP | `train.py` | `evaluate.py` | `hog_svm.py` | `compare_models.py` | `predict.py` | `report.py` | `results/` outputs | `models/` saved weights
