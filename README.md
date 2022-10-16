# CS550 Project - PICAI (Prostate Cancer) Challenge

## Data Preparation
Data preparation [script](./prepare_data.py) is used to convert the raw data to convert from MHA Archive to nnU-Net Raw Data Archive and also split the data into splits. This script is provided by Diagnostic Image Analysis Group for this challenge.

All the train, valid data-splits are created by [plan_overview.py](src/plan_overview.py) script, also provided by the Diagnostic Image Analysis Group.

## Data Interpretation
SimpleITK library is used to read and process the medical images for detection.

## Baseline Model - 
We used the monai framework to create the [Unet](https://en.wikipedia.org/wiki/U-Net) model.
The loss function and metrics are provided by the Grand Challenge.

### Loss function
- Focal Loss (Binary Segmentation loss)

### Metrics
- Average Precision (AP)
- Area Under the Receiver Operating Characteristic curve (AUROC)
- Overall AI Ranking Metric of the PI-CAI challenge: (AUROC + AP) / 2
- Precision-Recall (PR) curve
- Receiver Operating Characteristic (ROC) curve
- Free-Response Receiver Operating Characteristic (FROC) curve
