#!/usr/bin/python
from tf_unet import unet, util, image_util
import os
from os import path
import numpy as np

#preparing data loading
def prepare_data(data_root, img_glob):
    train_data = path.join(data_root, "train", img_glob)
    return image_util.ImageDataProvider(train_data)

#setup network
def setup_network():
    return unet.Unet(layers=2, features_root=64, channels=3, n_class=2)

#training
def train(data_provider, net):
    trainer = unet.Trainer(net)
    epochs = os.getenv("EPOCHS", "10")
    epochs = int(epochs)
    model_path = trainer.train(data_provider, "./unet_trained",
            training_iters=32, epochs=epochs)
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

def inspect_data(data_root):
    inspect_img_glob = os.getenv("INSPECT_GLOB", "*small.tif")
    data_provider = prepare_data(data_root,inspect_img_glob)
    ### INSPECT THE FREAKIN DATA
    np.set_printoptions(edgeitems=int(os.getenv("ROWS", "80")),
            linewidth=320,precision=2)
    x,y = data_provider(1)
    x = x.reshape(50,50,3)
    one_x = x[:,:,0]
    y = y.reshape(50,50,2)
    y = np.delete(y,1,axis=2)
    print(x.shape)
    print(y.shape)
    #print(x)
    print(np.matrix(one_x))
    print(np.matrix(y))

def main():
    net = setup_network()
    data_root = os.getenv("DATA_ROOT", os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "data"))
    if os.getenv("INSPECT") != None:
        inspect_data(data_root)
    if os.getenv("TRAIN") != None:
        train_img_glob = os.getenv("TRAIN_IMG_GLOB", "*.tif")
        data_provider = prepare_data(data_root, train_img_glob)
        model_path = train(data_provider, net)
    else:
        model_path = path.join(os.path.dirname(os.path.realpath(__file__)),
            "unet_trained", "model.ckpt")
    if os.getenv("VERIFY") != None:
        verify(data_root, net, model_path)

if __name__ == "__main__":
    main()
