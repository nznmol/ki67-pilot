# ki67 pilot project

We attempt to use machine learning methods for processing of WSIs (whole image
slides) in a breast cancer context.

We utilize the [tensor flow implementation](https://github.com/jakeret/tf_unet)
of the deep learning net
[unet](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

## Usage

Input for training is training data, while testing data is used for
verification after training.

All data is assumed to be located in `$DATA_ROOT` - training data in
`$DATA_ROOT/train/*.tif`, test data in `$DATA_ROOT/test/*.tif`.

For each image `$IMAGE.tif` in the training set, we expect to find an image
`$IMAGE_mask.tif` which is the annotated version of the image.

## Configuration

Config variables are set through the environment, default values given in
parantheses:

- `DATA_ROOT` (`$PWD/data`) path to data root directory
- `TRAIN` set if you want to train
- `VERIFY` set if you want to verify
- `EPOCHS` (10) number of epochs for training
- `FEATURES` (64) number of features
- `LAYERS` (2) number of layers
