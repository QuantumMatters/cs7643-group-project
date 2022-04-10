# Unsupervised Adversarial Image Reconstruction
OMSCS 7643 Deep Learning -- Group Project

# Running UNIR
Below are some instructions for running the UNIR source code, an implementation of the
unsupervised adversarial image reconstruction paper.

**Note**: There is a [fork](https://github.com/phosgene89/UNIR) of the UNIR project that might have addressed some of these steps.

1. Download CelebA dataset. Note that the official Google Drive link is typically down due to exceeding its daily usage. Try another source like https://www.kaggle.com/jessicali9530/celeba-dataset.
2. Clone the UNIR project [here](https://github.com/UNIR-Anonymous/UNIR).
3. Create an environment
    ```bash
    conda env create -f UNIR-environment.yml
    conda activate UNIR
    ```
4. Navigate to UNIR/unir/factory/dataset.py and set the value of line 19, 'filename': to the path to CelebA images.
5. Change all import statements from unsup_it to unir (the name of the directory containing the source code)
6. Edit line 46 in main.py to `use_mongo = False`
7. If running on a machine without a GPU, edit line 42 `device = "cpu"`


# Contributors
* Ethan Schaeffer
* Kasey Evans
* Duncan Wycliffe
* Mike Zhu
