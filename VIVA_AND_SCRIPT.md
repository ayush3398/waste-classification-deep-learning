# Viva answers and presentation script
Replace every [ ... ] with the real number from `results/RESULTS.md` after you run the code. Do not quote numbers you have not generated.

## 25 viva answers
1. **Why CNN?** Images have local patterns (edges, textures). A CNN learns filters that detect them anywhere in the image using few shared weights.
2. **Why not MLP?** An MLP flattens the image, losing neighbourhood structure, needs a weight for every pixel (49,152 inputs here), and overfits easily. We tested it: MLP test accuracy [ ]% vs CNN [ ]%.
3. **What is convolution?** Sliding a small filter over the image and computing a weighted sum at each position to make a feature map.
4. **What is a filter?** A small grid of learnable weights (3x3 here) that detects one pattern, e.g. an edge. We have 32, 64 and 128 filters in the three conv layers.
5. **What is ReLU?** f(x)=max(0,x). It adds non-linearity so the network can learn complex functions, and it is cheap and trains well.
6. **What is pooling?** Shrinking each feature map by summarising small regions. We use 2x2 max pooling, which halves width and height.
7. **Why max pooling?** It keeps the strongest response (is the feature present?), reduces computation and gives small translation tolerance.
8. **What is flattening?** Reshaping the 128x16x16 feature maps into one 32,768-long vector so dense layers can use it.
9. **Why dropout?** It randomly turns off half the units during training, so the network cannot depend on specific neurons. This reduces overfitting. It is off at test time.
10. **What is softmax?** It converts the 5 output scores into probabilities that add up to 1; the largest is the predicted class.
11. **Why CrossEntropyLoss?** It is the standard loss for multi-class classification: it punishes low probability on the correct class. In PyTorch it includes softmax, so the model outputs raw scores.
12. **What is backpropagation?** Using the chain rule to compute how much each weight contributed to the loss (its gradient), working backwards from the output.
13. **Where is backpropagation in PyTorch?** `loss.backward()` in `train.py`.
14. **Where are weights updated?** `optimizer.step()` (Adam) right after `loss.backward()`.
15. **What is overfitting?** The model memorises training data and performs worse on new data: training accuracy high, validation accuracy much lower.
16. **How did you reduce overfitting?** Data augmentation, dropout, and early stopping on validation loss (best-epoch weights restored).
17. **Why data augmentation?** Random flips, rotations and brightness changes create new variations of training images each epoch, so the model sees more variety without new photos. Only training data is augmented.
18. **Why this dataset?** TrashNet is public, widely used for waste classification, small enough for a one-hour laptop project and has real photos. It has no organic class, so I used 5 material classes (cardboard, glass, metal, paper, plastic).
19. **How did you split?** Stratified 70/15/15 (1670/358/359 images) with seed 42, done on file paths before any transformation; exact duplicates were removed first, and I checked no file or identical image is in two splits.
20. **What is your CNN's accuracy?** Test accuracy [ ]%, macro precision [ ], recall [ ], F1 [ ] on 359 unseen images (results/metrics.json).
21. **CNN vs SVM?** HOG+SVM got [ ]% vs CNN [ ]%. HOG uses fixed hand-made gradient features on grayscale; the CNN learns its own features including colour.
22. **CNN vs MLP?** MLP got [ ]% vs CNN [ ]%. [Explain using your actual numbers: if CNN is better, say convolution and weight sharing help; if not, say small data/limited epochs may explain it.]
23. **Your CO5 contribution?** Not a new algorithm: a regularised and augmented training strategy with an ablation study: baseline [ ]%, +augmentation [ ]%, +augmentation+dropout [ ]%. [State honestly which helped and which did not.]
24. **Limitations?** Small dataset, plain backgrounds, no organic class, near-duplicate photos of the same object may cross splits, one seed per experiment, cardboard is the smallest class.
25. **Future improvements?** Transfer learning (pretrained ResNet/MobileNet), more diverse data incl. organic waste, multiple seeds, split by object, Grad-CAM, real-world test photos.

## 2-3 minute presentation script (about 330 words)
**Problem.** Waste sorting is slow and often wrong. My project builds a deep learning model that looks at a photo of a waste item and tells its material: cardboard, glass, metal, paper or plastic.

**Dataset.** I used the public TrashNet dataset. After removing 3 exact duplicates I have 2,387 images in 5 classes. Every image is 512 by 384 pixels. The classes are fairly balanced; the biggest class, paper, is 1.47 times the smallest, cardboard. TrashNet has no organic class, so I did not claim one. I split the data 70, 15 and 15 percent into 1,670 training, 358 validation and 359 test images, and checked that no image appears in two splits.

**Preprocessing.** Each image is resized to 128 by 128 and normalised. Training images also get random flips, small rotations and brightness changes. Validation and test images are never augmented.

**CNN.** My network has three convolution and pooling blocks with 32, 64 and 128 filters, then a dense layer of 128 neurons, dropout, and an output layer with softmax for five probabilities. It has [ ] trainable parameters.

**Training.** I used Adam with learning rate 0.001, batch size 32, cross-entropy loss and early stopping. In each step the image goes forward through the CNN to a loss, loss.backward computes gradients, and optimizer.step updates the weights.

**Results.** On the unseen test set the CNN reached [ ] percent accuracy, with F1 score [ ]. The confusion matrix shows the most confusion between [ ] and [ ].

**Comparison.** On the same split, HOG plus SVM got [ ] percent and an MLP on flattened pixels got [ ] percent, so the CNN was [better/worse/similar] because [reason from your numbers].

**Innovation.** The CNN is not new. My contribution is a regularised, augmented training strategy with an ablation study. Baseline: [ ] percent. With augmentation: [ ]. With augmentation and dropout: [ ]. So [honest conclusion].

**Limitations and future work.** The dataset is small with plain backgrounds; in future I would use transfer learning and real-world images. Thank you.
