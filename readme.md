 **# Setting Up a Python Environment with Conda**

**Table of Contents**

* Creating a New Environment: #creating-a-new-environment
* Installing Packages: #installing-packages
* Running Jupyter Notebooks: #running-jupyter-notebooks
    * 0_prepare_data_create_mask_from_SPM12: #0_prepare_data_create_mask_from_spm12
    * 1_train_weak: #1_train_weak
    * 2_pred_img: #2_pred_img
    * 3_extract_feature_and_feature_selection_and_train_classify: #3_extract_feature_and_feature_selection_and_train_classify
    * 4_IOU_dice_score_w_doctor_: #4_iou_dice_score_w_doctor_

**Creating a New Environment**

1. **Install Conda:**
   - Download and install Conda from the official website: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

2. **Create the Environment:**
   - Open a terminal or command prompt and run:

     ```bash
     conda env create --file environment-nph.yml
     ```

3. **Activate the Environment:**
   - Run the following command:

     ```bash
     conda activate test-nph2
     ```

**Installing Packages**

1. **Install TensorFlow:**
   - Run the following commands:

     ```bash
     pip install --ignore-installed --upgrade tensorflow
     pip install tensorflow-addons==0.14.0
     ```

**Running Jupyter Notebooks**

     ```bash
     jupyter notebook
     ```
**0_prepare_data_create_mask_from_SPM12**

- Creates label masks from SPM12, classifying them into CSF, white matter, and grey matter using only confident 1.

**1_train_weak**

- Trains the segmentation model using the created label masks.
- Set the mask folder in the `create_generator_SI_fold` function:

    ```python
    mask_path = img_path.replace('/s/', '/segment_mask_confidence1_contour/')
    ```

**2_pred_img**

- Calls the segmentation model to generate predicted masks.
- Creates PNG files from each slice of the predicted mask.
- Converts the PNG files to .nii files.

**3_extract_feature_and_feature_selection_and_train_classify**

- Version (1): Loads pre-selected features and the best classification model.
- Version (2): Extracts features, performs feature selection, and trains a new classification model.
- Saves the trained model as `ClassifyModel_{current date}_{fold_index}.sav`.
- Evaluates the model and displays scores, including comparison with doctor's evaluation (P'Pun).

**4_IOU_dice_score_w_doctor_**

- Evaluates the predicted masks (.nii) against the doctor's labels (JSON format) for CSF only.
