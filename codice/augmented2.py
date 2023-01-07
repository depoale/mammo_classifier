"""similar to the tf tutorial
    but there is a bug in the newer tf versions so it doesn't work (too slow)
    might consider downgrading tf version"""

from keras import Sequential
from keras.layers import RandomFlip, RandomRotation
from matplotlib import pyplot as plt
from prova import get_data, get_model, plot 
from prova import callbacks,train_path, test_path, img_height, img_width, split, batch_size
train, val, test = get_data()

data_augmentation = Sequential(
    [
        RandomFlip("horizontal"),
        RandomRotation(0.5),
    ]
)

plt.figure(figsize=(10, 10))
for images, _ in train.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")