# How This Project Works — A Beginner's Guide

A ground-up explanation of the melanoma detection system. No prior machine learning
knowledge assumed. Each section only relies on what came before it.

---

## Contents

1. [The 30-second version](#the-30-second-version)
2. [Part 1 — The problem and the data](#part-1--the-problem-and-the-data)
3. [Part 2 — The model](#part-2--the-model)
4. [Part 3 — Training](#part-3--training)
5. [Part 4 — The threshold](#part-4--the-threshold)
6. [Part 5 — Measuring honestly](#part-5--measuring-honestly)
7. [Part 6 — Serving](#part-6--serving-from-a-saved-model-to-a-live-answer)
8. [Part 7 — The website](#part-7--the-website)
9. [Part 8 — MLflow, and the second big bug](#part-8--mlflow-and-the-second-big-bug)
10. [Part 9 — Configuration, testing, and shipping](#part-9--configuration-testing-and-shipping)
11. [Following one image all the way through](#following-one-image-all-the-way-through)
12. [The thread running through all of it](#the-thread-running-through-all-of-it)

---

## The 30-second version

Someone uploads a photo of a mole taken through a dermatoscope (a lit magnifying device a
dermatologist presses against the skin). The system returns a number between 0 and 1 — how likely it
thinks the mole is melanoma (skin cancer) — plus a heat-map showing which part of the image drove
that answer, plus a warning if the photo doesn't look like the kind of image it was trained on.

Everything else in this repository exists to make that one number **trustworthy**.

---

## Part 1 — The problem and the data

### What we're predicting

Melanoma is a skin cancer that looks, to an untrained eye, like an ordinary mole. Dermatologists
look at asymmetry, border irregularity, colour variation, size. The premise of this project: a
computer can learn those visual patterns from examples.

### The dataset

The data comes from **ISIC 2020**, a public competition dataset:

- **33,126 images**
- from only **2,056 patients**
- of which **584 are malignant** — that's **1.76%**

Those three numbers are the most important facts in the entire project. Almost every design decision
traces back to two of them:

**Fact A: only 2,056 patients for 33,126 images.** Each patient contributed a median of ~12 photos
of their *own* skin. Their moles all sit on the same skin tone, photographed with the same device,
under the same lighting.

**Fact B: 1.76% positive.** If the model just said "benign" every single time, it would be
**98.24% accurate**. Accuracy is a useless measurement here — this is why you'll see other metrics
later.

### Splitting the data — and why this is where projects die

Before you train anything, you divide the data into three piles:

| Pile | Purpose | Analogy |
|---|---|---|
| **train** (26,224 images) | The model learns from these | The textbook you study |
| **validation** (5,245) | Used *during* training to check progress and tune settings | Practice exams |
| **test** (1,657) | Touched only once, at the very end | The real final exam |

The test pile has to be genuinely unseen, or your final score is a lie.

Here's the trap. The obvious way to split is to shuffle all 33,126 images and deal them into three
piles. But remember Fact A — each patient has ~12 images. Shuffling randomly means patient #4021's
photos land in *all three piles*. The model sees their skin during training, then gets tested on more
of their skin.

That's like studying a practice exam that accidentally contains the real exam's questions. The model
can score well by recognising *that person's skin* rather than by recognising *melanoma*.

**This project originally had exactly that bug.** When it was measured: **1,656 of 1,657 test images
shared a patient with the training set**, including *all* the malignant ones. The reported score of
0.9355 was meaningless.

The fix is to split by **patient**, not by image — every photo from patient #4021 goes into exactly
one pile. That's what `StratifiedGroupKFold` does in `scripts/prepare_data.py`:

- "Group" = keep each patient's images together
- "Stratified" = keep roughly the same 1.76% malignant rate in each pile

The honest score after fixing this was **0.9116** instead of 0.9355. Lower — and real.

There's also an automated test (`tests/unit/test_patient_leakage.py`) that checks every patient ID
appears in only one pile. If anyone ever reintroduces the bug, the build fails. That's the real
lesson: *the fix isn't fixing the bug, it's making the bug impossible to reintroduce silently.*

### The image cache

Small practical thing that matters a lot. The original ISIC JPEGs are huge — up to 6000×4000 pixels.
Just *decoding* one costs about 0.3 seconds of CPU work.

The model only needs 384×384 pixels. If you decode a 6000×4000 image every time you show it to the
model (and you show it thousands of times), your expensive GPU sits idle waiting for the CPU. So
`scripts/resize_images.py` shrinks all 33,126 images once, into `data/processed/jpeg_448`, and
training reads from there. That's a ~20× speedup for a one-time cost.

---

## Part 2 — The model

### What a neural network is, roughly

A neural network is a very large mathematical function with millions of adjustable numbers, called
**weights**. You feed an image in one end and a prediction comes out the other. Training =
repeatedly nudging those millions of weights until the predictions get better.

You don't program the rules ("look for irregular borders"). You provide examples and the rules
emerge in the weights.

### The image branch: EfficientNet-B4

Rather than build one from scratch, this project uses **EfficientNet-B4** — a proven architecture —
that has already been trained on millions of general photographs (cats, cars, chairs). That's called
a **pretrained backbone**, and it's a huge head start: the early layers have already learned generic
vision skills like detecting edges, textures and colour patches. We just retrain it to care about
moles instead of cats. This is called **transfer learning**.

Fed a 384×384 image, EfficientNet-B4 outputs **1,792 numbers**. Those aren't pixels — they're an
abstract summary of the image, a "fingerprint". Think of it as the network's description of what it
sees: *"dark, asymmetric, blurred border, two colours"*, compressed into 1,792 dimensions.

### The metadata branch

Melanoma risk isn't only visual. Age, sex and body location genuinely matter. The dataset includes
those, so the project uses them.

Analysis in `notebooks/02_metadata_analysis.ipynb` confirmed those three fields alone predict better
than random guessing — that's what justified including them rather than assuming.

They go through a tiny network: 3 numbers → 64 → 32. Much smaller, because there's much less
information there.

### Fusion

Now you have two descriptions of the same case:

- 1,792 numbers from the image
- 32 numbers from the patient

Glue them together (1,824 numbers), pass through one more layer of 512, then squeeze down to
**a single number**.

```
Image (384×384) ──► EfficientNet-B4 ──► 1792 numbers ─┐
                                                       ├─► 1824 ─► 512 ─► 1 number
Age, sex, site ───► small MLP ────────► 32 numbers ───┘
```

That final single number is called a **logit** — it can be any value, like -3.2 or +1.7. A function
called the **sigmoid** squashes it into the range 0–1, which we read as a probability. -3.2 becomes
0.04; +1.7 becomes 0.85.

This "two inputs merged into one answer" design is why it's called **multimodal**.

---

## Part 3 — Training

### The basic loop

1. Take a batch of 16 images plus their metadata
2. Push them through the model → get 16 probabilities
3. Compare against the true answers → compute a **loss** (a single number measuring wrongness)
4. Compute which direction to nudge each weight to reduce the loss (**backpropagation**)
5. Nudge them slightly (the size of the nudge is the **learning rate**, here 0.0003)
6. Repeat

One full pass through all 26,224 training images is an **epoch**. This project runs 8.

**PyTorch** provides the maths. **PyTorch Lightning** provides the loop — the boilerplate for
batching, GPU handling, saving progress, and running validation checks. Without it you'd hand-write
several hundred lines of scaffolding that every project writes identically.

### Overfitting — the thing you're always fighting

Given enough capacity, a network will *memorise* the training images rather than learn general
rules. You can see it happen: training accuracy climbs toward perfection while validation accuracy
stalls or drops. It's the student who memorises the answer key instead of learning the subject.

Real numbers from this project's runs: the model peaked at epoch 2 with validation AUROC 0.922, then
by epoch 7 the *training* score had climbed to 0.995 while *validation* fell to 0.912. Memorising,
not learning.

Three defences:

**Early stopping** — stop training when validation stops improving.

**Checkpointing** — after each validation check, save a snapshot of the weights. Keep the best 3,
named by their score, e.g. `epoch=2-auroc=0.9220.ckpt`. At the end, you don't use the *final* model,
you use the *best* one. (This is where the project's second major bug lived — more shortly.)

**Augmentation** — every time an image is shown to the model, randomly distort it: flip it, rotate
it, shift the colours, blur it, punch small holes in it, re-compress it as a low-quality JPEG. The
model never sees exactly the same image twice, so memorising individual images becomes impossible;
it has to learn what melanoma actually looks like. (The JPEG and blur distortions are deliberate:
they teach it to survive images that weren't captured with ISIC's exact equipment.)

### The 1.76% problem

Remember Fact B. Out of a batch of 16 random images, on average **0.28** are malignant. Most batches
contain zero. The model learns almost nothing about the class you actually care about, and quickly
discovers that always guessing "benign" makes the loss very small.

Three layers of defence, and each one's exact setting was chosen because a different setting failed
visibly:

**1. Oversampling.** Rather than drawing images uniformly, draw malignant ones more often
(`WeightedRandomSampler`).

The obvious setting is 50/50 — half of every batch malignant. It doesn't work. There are only ~460
malignant training images; to fill half of every batch you'd show each one about **28 times per
epoch**, while ~40% of the benign images are never shown at all. The model memorises the small
positive set within the first epoch.

The project uses **15%** instead: positives still get ~8.5× emphasis, but ~86% of the negatives get
seen each epoch. (`positive_sample_rate: 0.15` in `configs/training/default.yaml`)

**2. Focal loss.** A standard loss treats all mistakes alike. Focal loss deliberately pays less
attention to examples the model already gets right, concentrating on the hard ones.

It has a parameter α that also re-weights classes. The famous paper that introduced focal loss uses
α=0.25, and copying that value is the default instinct. **It's wrong here** — the sampler has
*already* rebalanced the batches, so α=0.25 would correct for imbalance a second time and push every
predicted probability toward zero. This project uses **0.5**, which is neutral.
(`focal_alpha: 0.5`)

**3. Threshold calibration.** Explained next — it's important enough to get its own section.

---

## Part 4 — The threshold

The model outputs a probability like 0.31. To say "benign" or "malignant" you need a cut-off. The
instinctive choice is 0.5.

That's wrong, and here's why. Because malignant cases are rare, the model has learned that low
probabilities are usually right. Its scores are *ranked* well — malignant cases score higher than
benign ones — but they cluster low in absolute terms. Using 0.5 would mean calling almost everything
benign.

More importantly, **the two kinds of error are not equally bad**:

- **False positive**: healthy person told to see a doctor. Cost: anxiety, an appointment.
- **False negative**: melanoma called benign. Cost: potentially fatal.

So the threshold is chosen deliberately to catch at least **80% of melanomas**, accepting whatever
false-positive rate that requires. `src/cancer_detection/training/threshold.py` takes the best
checkpoint, scores every validation image, and finds the highest cut-off that still catches 80%. The
answer here: **0.3828**, saved to `artifacts/threshold.json`.

The subtle detail: it scores those validation images through *exactly the same pipeline the live
website uses* — including the 8-pass averaging described in Part 6. Calibrate on a slightly
different pipeline and the number you calculated isn't the number the deployed system honours.

---

## Part 5 — Measuring honestly

Run `scripts/evaluate.py` and the model faces the 1,657 test images it has never seen, from 102
patients it has never seen. Here's what comes out, and what each number means:

**AUROC = 0.9116.** "If I pick one random malignant image and one random benign image, how often
does the model score the malignant one higher?" 91% of the time. 0.5 would be coin-flipping; 1.0 is
perfect. This is the headline metric because it only cares about *ranking*, which sidesteps the
threshold question entirely.

**Sensitivity = 0.767.** Of the 30 actual melanomas, it caught 23 and missed 7. (Also called
recall.)

**Specificity = 0.837.** Of the 1,627 benign cases, it correctly cleared 1,362 and falsely flagged
265.

**PPV = 0.080.** *This is the number people find shocking.* Of the 288 cases it flagged as
malignant, only 23 were. **92% of its alarms are false.**

That is not a bug — it's arithmetic. When only 1.8% of cases are actually positive, there are 54×
more benign cases available to be mistakenly flagged. Even a good model produces mostly false alarms
at this prevalence. It's the same reason rare-disease screening tests always need confirmatory
follow-up. This project reports the number rather than hiding it.

**ECE = 0.177.** Measures whether the probabilities mean what they say — if the model says "0.30" a
hundred times, do 30 turn out malignant? 0.177 is poor. So the score should be read as a *ranking*
("this is more suspicious than that"), not as a literal probability of cancer.

**Confidence intervals.** AUROC's is 0.870–0.951. With only 30 malignant test images, one case going
the other way visibly shifts the numbers. The interval is honesty about that.

All of these live in `artifacts/test_metrics.json`, which is also what the website displays.

---

## Part 6 — Serving: from a saved model to a live answer

Training produces a file of weights. Getting from there to "a website anyone can use" is a separate
engineering problem with its own decisions.

### Test-time augmentation (TTA)

Instead of showing the model the uploaded image once, show it **8 times** — original, mirrored,
upside-down, rotated 90°/180°/270°, and two flip+rotate combinations. These 8 are the "dihedral
symmetries" of a square: all the ways to flip and rotate it that leave it square. Average the 8
probabilities.

Why: a mole is a mole regardless of orientation, so all 8 *should* agree. Averaging cancels random
noise, and the **spread** between them (`tta_std`) is itself informative — if the 8 views disagree
wildly, the model is unsure.

There was a bug here worth understanding. The original code used a library function called
`RandomRotate90`. Even when told "always apply this", it still *randomly picks* how many times to
rotate. So the same image uploaded twice returned different probabilities — and `tta_std` was
measuring the randomness of the augmentation rather than the uncertainty of the model. The fix was
to hard-code the 8 specific rotations (`_rot90` in `src/cancer_detection/data/transforms.py`). Now
identical uploads give identical answers, and `tta_std` means something.

Small optimisation: view #1 is the unmodified image, so it gets reused for the heat-map instead of
running a 9th pass.

### Aspect-ratio parity

A subtle one. ISIC images are roughly 3:2 (wider than tall). During training, images were randomly
cropped to squares. But validation and serving originally used a plain "resize to 384×384", which
*squashes* a 3:2 image horizontally.

So the model was trained on properly-proportioned moles and deployed on squashed ones. This mismatch
is invisible in every metric — because the metrics were computed with the same squashed transform.
Now both paths shrink the image until its shorter side is 384 and then crop the centre, preserving
proportions (`_to_square` in `data/transforms.py`).

The general lesson: **whatever you do to an image at training time must match what you do at serving
time.** Mismatches there are silent.

### Out-of-distribution detection

The model only knows dermoscopy images. Show it a selfie, an X-ray, or a photo of a dog and it will
still confidently return a probability — models don't know what they don't know.

So at startup, the API computes the 1,792-number fingerprint for a sample of training images and
works out the shape of that cloud of points. Any upload whose fingerprint sits far outside the cloud
(past the 99th percentile of a distance measure called **Mahalanobis distance**) gets flagged
`out_of_distribution: true`, and the interface warns you rather than confidently saying "benign".

Two engineering details in `src/cancer_detection/serving/ood.py` that show why this is harder than
it sounds:

- 1,792 dimensions is far too many to estimate a reliable cloud shape from a modest sample, so the
  data is first compressed to 64 dimensions (**PCA**).
- The 99th-percentile cut-off is measured on images *held out* from the ones used to build the
  cloud. Measuring it on the same images makes distances artificially small and the detector would
  then fire on ordinary inputs far more than the intended 1%.

### The heat-map

Users reasonably ask *why*. The system produces a heat-map overlay showing which image regions
pushed the prediction.

The standard technique is called GradCAM. It didn't work well here — it averages the importance
signal across the whole image, which produced hotspots that didn't sit on the lesion. The project
uses **HiResCAM**, which keeps the spatial detail. It also attaches to a specific internal layer
(`bn2`) rather than the very last one, because the last layer's edge cells produced a consistent
phantom hotspot in the top-right corner of every image.

There's a wiring problem too: the CAM technique expects a model that takes an image. This model
takes an image *and* metadata. So a small wrapper freezes the metadata and exposes an image-only
view, meaning the explanation reflects the image alone.

Important caveat, stated in the README: the heat-map shows what the model looked at. That is not the
same as showing where disease is.

### The API

**FastAPI** turns the Python model into something the internet can talk to. An "API" here is just a
URL that accepts a request and returns data:

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Send an image + age + sex + body site, get the JSON result back |
| `GET /health` | "Are you alive?" |
| `GET /test-metrics` | The honest test results, so the website can display them |
| `GET /metadata` | Which values the form accepts |

One design decision worth noting: loading the model takes a while. Originally the server refused to
answer *anything* until loading finished — which made automated health checks conclude the server
was dead and kill it. Now the server starts answering immediately, `/health` reports
`model_loaded: false` for the first few seconds, and prediction requests return "temporarily
unavailable" until the model is ready. **Degrade, don't crash.**

---

## Part 7 — The website

A **React** application (`frontend/`) written in TypeScript, built with Vite, styled with Tailwind.
It provides the upload form, shows the result card with the probability, label, confidence,
uncertainty and heat-map, and displays the model's honest test metrics.

Notably, those metrics aren't typed into the page by hand — they're read live from the
experiment-tracking system. The website can't drift out of sync with reality.

---

## Part 8 — MLflow, and the second big bug

### What MLflow does

Once you've run training 15 times with different settings, you need to know which run produced which
model and which score. **MLflow** is the record-keeper. Every run logs its settings, its metrics over
time, and the resulting model file.

It also acts as the handoff between training and serving. The rule this project uses: *the API
automatically loads the finished run with the highest validation score that saved a model.* No
copying files around, no pasting IDs.

That handoff depends on exact names — the model artifact must be called `model`, the metric must be
called `val/auroc`. Rename either and model loading breaks *silently*. That's why the README
documents them as fixed points.

### The bug

The original code did this at the end of training:

```python
mlflow.pytorch.log_model(lit_module.model)   # save the model
```

Looks right. It isn't. After training finishes, the model **sitting in memory** is the *last* epoch
— the most overfit one. The best checkpoint was saved to disk at epoch 2 and then forgotten.

So: the metrics reported epoch 2's performance, and the website served epoch 7's weights. Nothing
errored. Nothing looked wrong. The dashboard was describing a different model than the one answering
user requests.

`scripts/diagnose.py` measured the gap on held-out images:

| Model | Median score on true melanomas | Sensitivity |
|---|---|---|
| Best checkpoint (what metrics claimed) | 0.351 | 0.966 |
| What was actually being served | 0.201 | 0.759 |

The fix: explicitly reload the best checkpoint file from disk into a fresh model before logging it.

`diagnose.py` stays in the repo as a permanent audit tool. It scores four groups — training images
(how well it does on memorised data), held-out images (honest performance), those same held-out
images deliberately degraded to look web-sourced (does quality matter?), and any images you throw at
it. The remaining gap between the first two (median 0.914 vs 0.351) is the memorisation effect that
a leaked split would have concealed entirely.

---

## Part 9 — Configuration, testing, and shipping

### Hydra

Every setting — learning rate, batch size, which backbone, image size — lives in YAML files under
`configs/`, not in the code. That gives you:

```bash
python scripts/train.py                                    # defaults
python scripts/train.py model=resnet50 training.lr=1e-4    # override from the command line
python scripts/train.py -m model=efficientnet_b2,efficientnet_b4 training.lr=1e-3,5e-4  # all 4 combos
```

And critically, the full config is logged with each run — so six months later you can see exactly
what produced a given result.

### Tests

`tests/` contains automated checks that run on every code change. They use **fake generated data**,
so anyone can clone the repo and run the full test suite without downloading 12 GB or training
anything. The most valuable one is the patient-leakage test — it's a permanent guard against the
project's own worst bug.

```bash
pytest tests/unit -v
pytest tests/integration -v
```

### Docker

A container packages the code *plus* Python, PyTorch, system libraries, everything, into one
runnable image. It eliminates "works on my machine". Three containers here:

- **frontend** — nginx serving the React app
- **api** — FastAPI + PyTorch
- **mlflow** — the tracking server

`docker compose` runs all three together with one command. nginx also acts as a **reverse proxy**:
browsers only talk to port 3000, and nginx quietly forwards `/api` requests to the API and
`/mlflow` requests to MLflow. One address for the user; separate services underneath.

### AWS

- **EC2** — a rented Linux machine on the internet, running all three containers
- **ECR** — a private storage locker for the built container images
- **S3** — file storage, used once to migrate existing experiment history to the server

The deliberate split: **training happens on the local GPU laptop; only serving and record-keeping
run in the cloud.** GPU cloud instances are expensive and training is bursty; inference is cheap and
needs to be always-on. This mirrors how real teams operate.

### GitHub Actions (CI/CD)

Two automated pipelines:

**CI** runs on every code change: check formatting → check types → run unit tests → run integration
tests. If any step fails, the change is flagged. This is the safety net that makes the leakage test
meaningful — a test nobody runs protects nothing.

**CD** runs when you push to the main branch: build the three container images → push to ECR →
connect to the EC2 server → pull the new images → restart → poll until all three report healthy. No
manual deployment steps, no forgotten commands.

---

## Following one image all the way through

Ties it together:

1. You drop `mole.jpg` into the browser at `18.219.3.159:3000`, enter age 52, male, torso.
2. React packages it into a `POST` to `/api/predict`. nginx forwards to FastAPI.
3. FastAPI decodes the image. It shrinks the short side to 384 and crops the centre — same geometry
   as training.
4. The 1,792-number fingerprint is computed and compared against the training cloud. Not an outlier
   → `out_of_distribution: false`.
5. The image is rendered 8 ways (original + 7 flips/rotations). Each goes through EfficientNet-B4;
   the age/sex/site vector goes through the small MLP; they merge; 8 probabilities come out.
6. Average = 0.1847. Spread = 0.0312 (the 8 views agreed — good).
7. 0.1847 is below the calibrated threshold 0.3828 → **benign**. Confidence is measured as distance
   from *that* threshold, not from 0.5.
8. HiResCAM reuses view #1 to produce a heat-map, encoded as text so it can travel inside JSON.
9. JSON returns; React draws the result card, the heat-map overlay, the uncertainty, and the
   reminder that this isn't a medical device.

Elapsed: a second or two.

---

## The thread running through all of it

If you take one idea away, take this: **the hard part of machine learning is not making a number go
up. It's knowing whether the number is real.**

This project's three most instructive moments were all cases where everything looked fine and
wasn't:

- A test score of 0.9355 that measured memorised patients rather than learned medicine
- A dashboard describing epoch 2 while the website served epoch 7
- A "deterministic" augmentation that was quietly random, making the uncertainty estimate measure
  nothing

None of these threw an error. None showed up in any metric — because the metrics were computed with
the same broken assumptions. Each was found by deliberately going looking: measuring patient
overlap, scoring the served artifact against the checkpoint, uploading the same image twice.

That's why the repository contains a leakage test, a diagnostic script, an OOD detector and an
explicit PPV of 0.080 in the README. The model is ordinary. The scepticism is the work.

---

## Where to look next

| If you want to understand… | Read |
|---|---|
| How the data gets split | `scripts/prepare_data.py` |
| The model architecture | `src/cancer_detection/models/classifier.py` |
| Augmentation and the geometry fixes | `src/cancer_detection/data/transforms.py` |
| The training loop and loss | `src/cancer_detection/training/lit_module.py`, `losses.py` |
| Threshold calibration | `src/cancer_detection/training/threshold.py` |
| Prediction, TTA, confidence | `src/cancer_detection/serving/predictor.py` |
| The OOD detector | `src/cancer_detection/serving/ood.py` |
| The web API | `src/cancer_detection/serving/api.py` |
| All the settings | `configs/` |
| The summary of everything | [`../README.md`](../README.md) |
