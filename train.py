from tf_unet import unet, util, image_util
import os
import path

#preparing data loading
data_root = os.getenv("DATA_ROOT")
train_data = path.join(data_root, "train", "*.tif")
data_provider = image_util.ImageDataProvider(train_data)

#setup & training
net = unet.Unet(layers=3, features_root=64, channels=3, n_class=2)
trainer = unet.Trainer(net)
path = trainer.train(data_provider, "./unet_trained", training_iters=32, epochs=100)

#verification

test_data = path.join(data_root, "test", "*.tif")
test_data_provider = image_util.ImageDataProvider(test_data)

x_test, _ = test_data_provider(1)

prediction = net.predict(path, x_test)

#unet.error_rate(prediction, util.crop_to_shape(label, prediction.shape))

img = util.combine_img_prediction(x_test, x_test, prediction)
util.save_image(img, "prediction.jpg")
