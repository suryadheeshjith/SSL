# SSL

Self supervised learning on videos!


## Code Structure

The code is divided into 4 main files: 

`FJEPA.py` - contains the implementation for the FJEPA model

`downstream.py` - contains the implementation of the downstream UNet model used for mask segmentation

`train.py` - file to run model training

`eval.py` - contains the code required to load the models and evaluate the final predictions

## Model Training

To run training code, simply execute `python train.py`. To run training on the HPC server, run the `train.SBATCH` script. 

By default, the script trains the `FJEPA` model. To train the `UNet` model, uncomment the code below the relevant comment. You can also rename both model names as desired for the `.pth` file, and the best model weights are stores as `best_{model_name}.pth`.

## Model evaluation

The `eval.py` file loads the models as specified in the `model_name` and `PATH` variables. The script then computes the Jaccard Index score over the `val` dataset. 

To run the evaluation on the HPC server, run the `eval.SBATCH` script.

