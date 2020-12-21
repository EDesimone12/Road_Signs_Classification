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
from keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
from sklearn.model_selection import train_test_split


tf.random.set_seed(100)

path='../resources/annotations/'
content=[]
speedcounter = 0

for filename in os.listdir(path):

    if not filename.endswith('.xml'): continue
    finalpath = os.path.join(path, filename)
    print(finalpath)

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
#data.head()

print("Waiting. . .")
data1=[]

i=0

for a in data.path_name.values:

    image = Image.open("/content/drive/MyDrive/FIA_Segnali_Stradali/"+a).convert("RGB")

    #Image resizing is needed to upgrade the resolution
    image=image.resize((224,224),Image.ANTIALIAS)
    image=np.array(image.getdata()).reshape(224,224,3)
    data1.append(image)

print("---Done---")

X=np.array(data1)

y=np.array(data.iloc[:,-1],dtype=int)

c=to_categorical(y,dtype=int)
Y=c[:,1:]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=787)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#Layer definition
model = Sequential()
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=X_train.shape[1:]))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(rate=0.50))
model.add(Dense(4, activation='softmax'))

#Compilation of the model
#categorical_crossentropy(multiclass classification problems)
#Optimizer Adaptive Moment lr=0.001
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(X_train, y_train, batch_size=8, epochs=50, validation_data=(X_test, y_test))

#Evaluating
results= model.evaluate(X_test,y_test,batch_size=8)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()