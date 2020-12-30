import bs4
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
from sklearn.model_selection import train_test_split

tf.random.set_seed(100)

path = "..\\resources\\annotations"
content = []
speedcounter = 0

for filename in os.listdir(path):

    if not filename.endswith('.xml'): continue
    finalpath = os.path.join(path, filename)

    infile = open(finalpath, "r")

    contents = infile.read()
    # Parsing xml
    soup = bs4.BeautifulSoup(contents, 'xml')
    class_name = soup.find_all("name")
    name = soup.find_all('filename')
    width = soup.find_all("width")
    height = soup.find_all("height")
    depth = soup.find_all("depth")

    ls = []
    for x in range(0, len(name)):
        for i in name:
            name = name[x].get_text()
            path_name = "images/" + name

        class_name = class_name[x].get_text()
        if class_name == 'speedlimit':
            if speedcounter < 75:
                # It counts the 'speedlimit' images
                height = int(height[x].get_text())
                depth = int(depth[x].get_text())
                width = int(width[x].get_text())
                f_name = filename
                ls.extend([f_name, path_name, width, height, depth, class_name])
                speedcounter = speedcounter + 1
                content.append(ls)
        else:
            # It selects the other classes('stop','crosswalk','trafficlight')
            height = int(height[x].get_text())
            depth = int(depth[x].get_text())
            width = int(width[x].get_text())
            f_name = filename
            ls.extend([f_name, path_name, width, height, depth, class_name])
            content.append(ls)

new_cols = ["f_name", "path_name", "width", "height", "depth", "class_name"]
data = pd.DataFrame(data=content, columns=new_cols)
data.class_name = data.class_name.map({'trafficlight': 1, 'speedlimit': 2, 'crosswalk': 3, 'stop': 4})
print(data.shape)
# data.head()

print("Waiting. . .")
data1 = []

i = 0

for a in data.path_name.values:
    image = Image.open("..\\resources\\" + a).convert("RGB")

    # Image resizing is needed to upgrade the resolution
    image = image.resize((224, 224), Image.ANTIALIAS)
    image = np.array(image.getdata()).reshape(224, 224, 3)
    data1.append(image)

print("---Done---")

X = np.array(data1)

y = np.array(data.iloc[:, -1], dtype=int)

c = to_categorical(y, dtype=int)
Y = c[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=787)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Layer definition
model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:]))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(rate=0.50))
model.add(Dense(4, activation='softmax'))

# Compilation of the model
# categorical_crossentropy(multiclass classification problems)
# Optimizer Adaptive Moment lr=0.001
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(X_train, y_train, batch_size=8, epochs=50, validation_data=(X_test, y_test))

# Evaluating
results = model.evaluate(X_test, y_test, batch_size=8)

#Saving final model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to drive")

# ***************************************************************************
# We have tested the trained model.                                         *
# In this code the Neural Network can classified the road sign we submit.   *
# This will be available in the demo.                                       *
# ***************************************************************************

# ** load json and create model **
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
# ** load weights into new model **
#loaded_model.load_weights("model.h5")
#print("Loaded model from disk")
#
# ** load images **
#path = "/content/drive/MyDrive/FIA_SegnaliStradali/toPredict/predict"
#
#print("Insert a class to detect between: \n crosswalk \n limit \n stop \n trafficlight")
#inputX = input()
#path = path + inputX +".png"
#
#immagine = Image.open(path).convert("RGB")
#
# ** Image formatting **
#img=immagine.resize((224,224),Image.ANTIALIAS)
#img=np.reshape(img,[224,224,3])
#
#plt.imshow(img)
#plt.show()
#
# ** predict **
#img_array = image.img_to_array(img)
#img_batch = np.expand_dims(img_array, axis=0)
#
#img_preprocessed = preprocess_input(img_batch)
#
#prediction = loaded_model.predict(img_preprocessed)
#print(prediction)
#
# ** Generate arg maxes for predictions **
#classes = np.argmax(prediction, axis = 1)
#print("trafficlight(0), speedlimit(1), crosswalk(2), stop(3)")
#print(classes)

# *******************************************************************************************
# Here ends the demo.                                                                       *
# *******************************************************************************************

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
