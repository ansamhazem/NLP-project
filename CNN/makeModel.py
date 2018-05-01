import json
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation , Flatten
from sklearn.model_selection import train_test_split
from keras import optimizers


news = pd.read_csv("..\data\uci-news-aggregator.csv") #Importing data from CSV

news = (news.loc[news['CATEGORY'].isin(['b','e','t','m'])]) #Retaining rows that belong to categories 'b' and 'e'
news=news[0:10000]

X_train, X_test, Y_train, Y_test = train_test_split(news["TITLE"], news["CATEGORY"], test_size = 0.2)
x_train = np.array(X_train);
x_test = np.array(X_test);
y_train = np.array(Y_train);
y_test = np.array(Y_test);


tokenizer = Tokenizer(num_words=10000)
print("tttttttttttttttttttttt")
print(tokenizer)

# feed our tweets to the Tokenizer
tokenizer.fit_on_texts(x_train)
tokenizer.fit_on_texts(x_test)
print("tareeeeeeeek")

# Tokenizers come with a convenient list of words and IDs
dictionary = tokenizer.word_index
# Let's save this out so we can use it later
with open('dictionary.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)


def convert_text_to_index_array(text):
    # one really important thing that `text_to_word_sequence` does
    # is make all texts the same length -- in this case, the length
    # of the longest text in the set.
    return [dictionary[word] for word in kpt.text_to_word_sequence(text)]

allWordIndices = []
# for each tweet, change each token to its ID in the Tokenizer's word_index
for text in x_train:
    wordIndices = convert_text_to_index_array(text)
    allWordIndices.append(wordIndices)

# now we have a list of all tweets converted to index arrays.
# cast as an array for future usage.
allWordIndices = np.asarray(allWordIndices)

# create one-hot matrices out of the indexed tweets
x_train = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')



allWordIndices = []
for text in x_test:
    wordIndices = convert_text_to_index_array(text)
    allWordIndices.append(wordIndices)

# now we have a list of all tweets converted to index arrays.
# cast as an array for future usage.
allWordIndices = np.asarray(allWordIndices)

# create one-hot matrices out of the indexed tweets
x_test = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
# treat the labels as categories
yy_train=[]
yy_test=[]
  #[e,t,b,m]
for y in y_test:    
    if(y=='m'):
#          label=np.zeros(4)
#          label[3] = 1.0
        yy_test.append(3)
    if(y=='e'): 
#          label=np.zeros(4)
#          label[0] = 1.0
        yy_test.append(0)
    if(y=='t'): 
#          label=np.zeros(4)
#          label[1] = 1.0
        yy_test.append(1)
    if(y=='b'): 
#          label=np.zeros(4)
#          label[2] = 1.0
        yy_test.append(2)
        
        
        
for y in y_train:
    

    if(y=='m'):
#          label=np.zeros(4)
#          label[3] = 1.0
        yy_train.append(3)
    if(y=='e'): 
#          label=np.zeros(4)
#          label[0] = 1.0
        yy_train.append(0)
    if(y=='t'): 
#          label=np.zeros(4)
#          label[1] = 1.0
        yy_train.append(1)
    if(y=='b'):
#          label=np.zeros(4)
#          label[2] = 1.0
        yy_train.append(2)
 
yy_train=np.array(yy_train)  
yy_test=np.array(yy_test)

#yy_train = keras.utils.to_categorical(yy_train, 4)
    

#(train_y.index(train_y)


sequence_length = x_train.shape
print("ttttttttttttt",yy_train.shape)
print("yyyyyyyyggggggggggggggggggyyyyyy",x_train.shape)

input_tensor=(8000,)

model = Sequential()
model.add(Dense(256, input_shape=input_tensor, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.summary()

model.compile(loss='binary_crossentropy',
    optimizer='sgd',
    metrics=['accuracy'])

#print(x_train.shape)
model.fit(x_train, yy_train,
    batch_size=32,
    epochs=10,
    verbose=1,
    validation_data=(x_test,yy_test)
    )

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model2.h5')

print('saved model!')
