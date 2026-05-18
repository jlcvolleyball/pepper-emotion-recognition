# Emotion Recognition During Exercise for Older Adults
## Overview
This project was developed as part of ongoing research in creating a robotic exercise coach aimed to improve motivation for physical activity in aging adults.
To experiment with introducing emotional intelligence to the exercise coach, our first step was to develop a model that could perform emotion recognition using 
input data (ex: visual, audio, etc.). However, existing off-the-shelf models underperform for our target population (older adults) because they are trained
on general datasets of all age groups. Thus, the objective of this project is to train and evaluate an emotion recognition ML model tailored towards older adults.

To this end, three iterations of models were trained.

### Model 1: Two-Stage Training
A model that takes in as input an image (or a video frame) and outputs a classification. Model 1 was fine-tuned in two stages from the pretrained ResNet-50. 
The first stage involved training the architecture on the full FER2013 dataset to gain familiarity with the emotion classification task, and the second stage
involved further fine-tuning on a subset of FER2013 with only older adults of age 50+.
<img width="797" height="415" alt="Screenshot 2026-05-18 at 3 27 43 PM" src="https://github.com/user-attachments/assets/9ef4110d-8991-4348-8888-995e9913fa43" />


### Model 2: Video-Only Model
A model that takes in as input a video and outputs a valence score from 1 (very negative emotion) to 7 (very positive emotion). Model 2 was trained on videos from 
the ElderReact dataset.

### Model 3: Video+Audio (Multimodal) Model
A model that takes in as input a video and outputs a valence score from 1 (very negative emotion) to 7 (very positive emotion). Model 2 was trained on videos and their audio
from the ElderReact dataset.
<img width="843" height="477" alt="Screenshot 2026-05-18 at 3 27 23 PM" src="https://github.com/user-attachments/assets/ea59c5a8-9567-4dc1-81d0-6fa3a7165b6b" />


See the training scripts for each of these models for more details on their training processes. 

---

## Organization
```
.
├── data/
│   ├── raw/
│   └── processed/
├── artifacts/
├── inference_outputs/
│   ├── video_only/
│   └── video_audio/
├── scripts/
│   └── fer2013_default_pipeline.py
├── src/
│   ├── create_test_subset.py
│   ├── filter_fer2013_age.py
│   ├── metric_compare_elderreact.py
│   ├── preproess_elderreact.py
│   ├── preprocess_fer2013.py
│   ├── run_elderreact.py
│   ├── train_elderreact.py
│   ├── train_fer2013.py
│   ├── train_finetune_fer2013.py
│   ├── train_multimodal_sweep.py
│   └── util.py
```

### Note on directories:
Apart from the data directory in the project structure above, all other directories are created after a script is run. In order
to run any of the training scripts (discussed below), create a directory `data/raw/` under the root directory, and copy your desired
dataset insider of it.

### Training Scripts:
The training script for Model 1 is `src/train_finetune_fer2013.py`. For Model 1, a training pipeline was written, `scripts/fer2013_default_pipeline.py`.
The training script for Models 2 and 3 is `src/train_elderreact.py`. Note that Models 2 and 3 were also trained using wandb hyperparameter sweeps, which 
can be found in `src/train_multimodal_sweep.py`. Other training scripts are present for record-keeping, but are not directly used in training for Model 1 
and Model 2.

### Inference Scripts:
The inference script for Models 2 and 3 is `src/run_elderreact.py`.

### Evaluation Scripts:
The evaluation is performed with training for all models. Post-evaluation for further comparison between Models 2 and 3 is performed in `src/metric_compare_elderreact.py`.

### Other:
Involved in preprocessing, general testing, or util.

---

## Run Pipeline

### Model 1
1. Copy FER2013 dataset into data directory as instructed above
2. Run age extraction preprocessing (NOTE: due to the formatting of the FER2013 dataset, it may be necessary to run this script twice--once for train, and once for test. To support this,
   the script allows for arguments, which can be directly edited in the script or passed in through the terminal, but are not provided here due to differences in naming
   conventions)
```bash
python src/filter_fer2013_age.py
```
3. Run validation extraction preprocessing
```bash
python src/preprocess_fer2013.py
```
4. Run training
```bash
python src/train_finetune_fer2013.py
```
Because a pipeline script for training Model 1 is provided in scripts, it is also possible to simply run this script instead. However, note step 2 of the instructions above
and ensure that the arguments are correct before running the pipeline.
```bash
python scripts/fer2013_default_pipeline.py
```

### Models 2 and 3
1. Copy ElderReact dataset into data directory as instructed above
2. Run preprocessing
```bash
python src/preproess_elderreact.py
```
3. Run training (either once for Models 2 and 3 or with the hyperparameter sweep for Model 3).
Once:
```bash
python src/train_elderreact.py
```
Hyperparameter sweep for Model 3:
```bash
python src/train_multimodal_sweep.py
```
4. Run evaluation (if desired)
```bash
python src/metric_compare_elderreact.py
```
5. Run inference (if desired)
```bash
python src/run_elderreact.py
```
