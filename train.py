#!/usr/bin/python
from tf_unet import unet, util, image_util
import os
from os import path
import tensorflow as tf
from functools import reduce
import numpy as np
from scipy import misc
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

#count variables
def count_params():
    "print number of trainable variables"
    size = lambda v: reduce((lambda x, y: x*y), v.get_shape().as_list())
    n = sum(size(v) for v in tf.trainable_variables())
    logging.info("Model size: %dk" % (n/1000,))

def count_relevant_pixels():
    data_root = os.getenv("DATA_ROOT", os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "data"))
    train_data = path.join(data_root, "train", "*_mask.tif")
    pixels = 0
    for img_path in sorted(glob.glob(train_data)):
        file = path.basename(img_path)
        mask = np.fromfile(img_path, dtype=np.bool)
        nonzero_indices = mask.nonzero()
        ratio = 1 - len(np.flatnonzero(mask))/float(mask.size)
        pixels = 768*768*ratio
    logging.info("Relevant pixels in training data: %dk " % (pixels/1000))

def count():
    count_params()
    count_relevant_pixels()

#preparing data loading
def prepare_data(data_root):
    train_data = path.join(data_root, "train", "*.tif")
    return image_util.ImageDataProvider(train_data)

#setup network
def setup_network(layers, features_root):
    return unet.Unet(layers=layers, features_root=features_root, channels=3, n_class=2)

#training
def train(data_provider, net):
    trainer = unet.Trainer(net, batch_size=1, verification_batch_size=4)
    epochs = int(os.getenv("EPOCHS", "10"))
    display_step = int(os.getenv("DISPLAY_STEP", "1"))
    restore_model = True if os.getenv("RESTORE_MODEL") else False
    model_path = trainer.train(data_provider, "./unet_trained",
            training_iters=32, epochs=epochs, display_step=display_step,
            restore=restore_model)
    return model_path

#verification
def verify(data_root, net, model_path):
    test_data = path.join(data_root, "test", "*.tif")
    test_data_provider = image_util.ImageDataProvider(test_data)
    x_test, _ = test_data_provider(1)
    prediction = net.predict(model_path, x_test)
    img = util.combine_img_prediction(x_test, x_test, prediction)
    util.save_image(img, "prediction.jpg")

#unet.error_rate(prediction, util.crop_to_shape(label, prediction.shape))

def main():
    layers = int(os.getenv("LAYERS", "2"))
    features = int(os.getenv("FEATURES", "64"))
    net = setup_network(layers, features)
    data_root = os.getenv("DATA_ROOT", os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "data"))
    count()
    if os.getenv("TRAIN") != None:
        data_provider = prepare_data(data_root)
        model_path = train(data_provider, net)
    else:
        model_path = path.join(os.path.dirname(os.path.realpath(__file__)),
            "unet_trained", "model.ckpt")
    if os.getenv("VERIFY") != None:
        verify(data_root, net, model_path)

if __name__ == "__main__":
    main()
