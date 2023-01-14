import cv2 
import os
import sys
from PIL import Image
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../data_opt"))
) 
from models import get_data, get_model, plot 
from models import callbacks, img_height, img_width, split, batch_size
train_path = 'data_opt/Train'
test_path = 'data_opt/Test'

""" #used to save images with expanded dynamic range
for file in os.listdir('data_png/Train/0'):
    filename, extension  = os.path.splitext(file)

    img = cv2.imread(os.path.join('data_png/Train/0',file),0)
    equ = cv2.equalizeHist(img)
    #res = np.hstack((img,equ)) #stacking images side-by-side
    cv2.imwrite(os.path.join('data_opt/Train/0',file),equ) """

train, val, test = get_data(train_path=train_path, test_path=test_path)
model = get_model()
history = model.fit(train, batch_size=batch_size , epochs=500, validation_data=val, callbacks=callbacks)

plot(history=history)
print(f'test accuracy: {round(model.evaluate(test)[1],3)}')
 