# Astronomy pipeline
This application predicts whether the isochrone is matching, using the following metrics:
 - Gaia G magnitude
 - Gaia GBP magnitude
 - Gaia GBP magnitude
 - extinction parameter
 - extinction in the V band
 - extinction in the G band
 - extinction in the BP band
 - extinction in the RP band
 - color excess in BP-RP color
 - color excess in B-V color
 - V-I color without reddening
 - V-I color with reddening
 - Absolute magnitude in the Gaia G band
 - Stellar radius
 - Linear distance of the source
 - isochrone matching flag (0 or 1)
 - Stellar effective temperature
 - Stellar gravity
 - Stellar [Fe/H] abundance ratio
 - Stellar [alpha/Fe] abundance ratio

## Dependencies
 - Python 3
 - Tensorflow 2
 - Pandas
 - Numpy

Optional:
 - Cuda: only if the model has to run on the gpu

## Running the application
Use `flask --app runner.py run` to run the server.
Use `python trainer.py [dataset_path]` to train a new model, replace `[dataset_path]` with the actual path to the dataset.
Use `python best_label.py [dataset_path]` to find the best label for training the model, replace `[dataset_path]` with the actual path to the dataset.

## Model
### Shape
| Layer (type) | Output Shape | Param # |  
|-|-|-|
| dense (Dense) | (None, 4) | 76 |
| dense_1 (Dense) | (None, 2) | 10 |
| dense_2 (Dense) | (None, 1) | 3 |

Total params: 89
Trainable params: 89
Non-trainable params: 0

### Results on datasets
#### [parameters_MARCS_LIBRARY](dataset/parameters_MARCS_LIBRARY.csv)
Training loss: 0.2136
Training accuracy: 0.9140

Testing loss: 0.1994
Testing accuracy: 0.9197
