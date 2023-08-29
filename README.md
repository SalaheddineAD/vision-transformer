# vision-transformer
## Setup
```Run these commands:
conda create -n "vision-transformer" python==3.8 -y
conda activate vision-transformer
pip install -r requirements.txt
For now use only notebook not src folder
```
## Project Organization
### notebooks: This directory holds Jupyter notebooks .
* They can only be run on collab

### src: This directory contains your project's source code, organized by functionality.

* data: Functions/classes for loading, preprocessing, and augmenting data.
* models: Model architectures, training loops, and related functions.
* utils: General utility functions that might be used across different parts of the project.
* main.py: A script that orchestrates the entire pipeline, from data processing to model training and evaluation.

### experiments: Each subdirectory corresponds to an individual experiment you run.

* experiment_X: Contains the assets for a specific experiment.
* config.yaml: Configuration file capturing hyperparameters and settings for this experiment.
* logs: Directory for storing training logs, such as loss and accuracy curves.
* models: Directory for saving trained models or checkpoints.
### results: This directory stores the results and outputs of your experiments.

* experiment_X: Contains the results of a specific experiment.
* metrics: Metrics calculated during training and evaluation.
* plots: Visualizations and plots generated during the experiment.
