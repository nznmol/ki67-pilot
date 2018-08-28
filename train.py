#!/usr/bin/python
from tf_unet import unet, util, image_util
import os
from os import path

#preparing data loading
def prepare_data(data_root):
    train_data = path.join(data_root, "train", "*.tif")
    return image_util.ImageDataProvider(train_data)

#setup network
def setup_network():
    return unet.Unet(layers=3, features_root=64, channels=3, n_class=2)

#training
def train(data_provider, net):
    trainer = unet.Trainer(net)
    model_path = trainer.train(data_provider, "./unet_trained",
            training_iters=32, epochs=25)
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
    net = setup_network()
    data_root = os.getenv("DATA_ROOT", os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "data"))
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
