#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import argparse
import json
import os
from pathlib import Path

import SimpleITK as sitk
from picai_prep import MHA2nnUNetConverter

"""
Script to prepare PI-CAI data into the nnUNet raw data format
For documentation, please see:
https://github.com/DIAGNijmegen/picai_baseline#prepare-data
"""

# set paths
parser = argparse.ArgumentParser()
parser.add_argument("--workdir", type=str, default=os.environ.get("workdir", "/workdir"),
                    help="Path to the working directory (default: /workdir, or the environment variable 'workdir')")
parser.add_argument("--inputdir", type=str, default=os.environ.get("inputdir", "/input"),
                    help="Path to the input dataset (default: /input, or the environment variable 'inputdir')")
parser.add_argument("--imagesdir", type=str, default="images",
                    help="Path to the images, relative to --inputdir (default: /input/images)")
parser.add_argument("--labelsdir", type=str, default="picai_labels",
                    help="Path to the labels, relative to --inputdir (root of picai_labels) (default: /input/picai_labels)")
args, _ = parser.parse_known_args()

# parse paths
workdir = Path(args.workdir)
inputdir = Path(args.inputdir)
imagesdir = Path(inputdir / args.imagesdir)
labelsdir = Path(inputdir / args.labelsdir)

# settings
task = "Task2201_picai_baseline"

# paths
annotations_dir = labelsdir / "csPCa_lesion_delineations/human_expert/resampled/"
mha2nnunet_settings_path = workdir / "mha2nnunet_settings" / "Task2201_picai_baseline.json"
nnUNet_raw_data_path = workdir / "nnUNet_raw_data"
nnUNet_task_dir = nnUNet_raw_data_path / task
nnUNet_dataset_json_path = nnUNet_task_dir / "dataset.json"
nnUNet_splits_path = nnUNet_task_dir / "splits.json"


def preprocess_picai_annotation(lbl: sitk.Image) -> sitk.Image:
    """Binarize the granular ISUP ≥ 2 annotations"""
    lbl_arr = sitk.GetArrayFromImage(lbl)

    # convert granular PI-CAI csPCa annotation to binary csPCa annotation
    lbl_arr = (lbl_arr >= 1).astype('uint8')

    # convert label back to SimpleITK
    lbl_new: sitk.Image = sitk.GetImageFromArray(lbl_arr)
    lbl_new.CopyInformation(lbl)
    return lbl_new

# read preprocessing settings and set the annotation preprocessing function
with open(mha2nnunet_settings_path) as fp:
    mha2nnunet_settings = json.load(fp)

if not "options" in mha2nnunet_settings:
    mha2nnunet_settings["options"] = {}
mha2nnunet_settings["options"]["annotation_preprocess_func"] = preprocess_picai_annotation

if not "preprocessing" in mha2nnunet_settings:
    mha2nnunet_settings["preprocessing"] = {}

mha2nnunet_settings["preprocessing"]["spacing"] = [3.0, 0.5, 0.5]
mha2nnunet_settings["preprocessing"]["matrix_size"] = [20, 256, 256]

# prepare dataset in nnUNet format
archive = MHA2nnUNetConverter(
    output_dir=nnUNet_raw_data_path,
    scans_dir=imagesdir,
    annotations_dir=annotations_dir,
    mha2nnunet_settings=mha2nnunet_settings,
)

archive.convert()
archive.create_dataset_json()

print("Finished.")
