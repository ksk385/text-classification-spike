from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

import argparse

parser = argparse.ArgumentParser(description='Text Classification.')
parser.add_argument('--verbose', action='store_true',
                   help='displays debug logging')
parser.add_argument('--feature', choices=['count', 'tfidf_word'],
                    default='count')

args = parser.parse_args()
print(args)

# load the dataset
# sample data
# __label__2 an absolute masterpiece: I am quite sure any of you actually taking the time to read this have played the game at least once, and heard at least a few of the tracks here. And whether you were aware of it or not, Mitsuda's music contributed greatly to the mood of every single minute of the whole game.Composed of 3 CDs and quite a few songs (I haven't an exact count), all of which are heart-rendering and impressively remarkable, this soundtrack is one I assure you you will not forget. It has everything for every listener -- from fast-paced and energetic (Dancing the Tokage or Termina Home), to slower and more haunting (Dragon God), to purely beautifully composed (Time's Scar), to even some fantastic vocals (Radical Dreamers).This is one of the best videogame soundtracks out there, and surely Mitsuda's best ever. ^_^
# __label__1 Buyer beware: This is a self-published book, and if you want to know why--read a few paragraphs! Those 5 star reviews must have been written by Ms. Haddon's family and friends--or perhaps, by herself! I can't imagine anyone reading the whole thing--I spent an evening with the book and a friend and we were in hysterics reading bits and pieces of it to one another. It is most definitely bad enough to be entered into some kind of a "worst book" contest. I can't believe Amazon even sells this kind of thing. Maybe I can offer them my 8th grade term paper on "To Kill a Mockingbird"--a book I am quite sure Ms. Haddon never heard of. Anyway, unless you are in a mood to send a book to someone as a joke---stay far, far away from this one!
data = open('data/corpus').read()
labels, texts = [], []
for i, line in enumerate(data.split('\n')):
    content = line.split()
    labels.append(content[0])
    texts.append(content[1:])
if args.verbose:
    print(f'Raw Split Data')
    print(f'*******')
    print(f'Labels: {labels[0:2]}')
    print(f'Texts: {texts[0:2]}')
    print(f'*******')

# create a data frame using texts and labels
trainDF = pandas.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels

if args.verbose:
    print(f'Data Frame')
    print(f'*******')
    print(trainDF.values)
    print(f'*******')

# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])
if args.verbose:
    print(f'Train X: {train_x}')
    print(f'*******')
    print(f'Valid X: {valid_x}')
    print(f'*******')
    print(f'Train Y: {train_y}')
    print(f'*******')
    print(f'Valid Y: {valid_y}')
    print(f'*******')

# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

if args.verbose:
    print(f'Label Encoding')
    print(f'*******')
    print(f'Labels Encoded Train Y: {train_y}')
    print(f'Labels Encoded Valid Y: {valid_y}')
    print(f'*******')

if args.feature == 'count':
    # create a count vectorizer object 
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    array_of_texts = [" ".join(sentence_array) for sentence_array in trainDF.text.values]
    array_of_training_texts = [" ".join(sentence_array) for sentence_array in train_x]
    array_of_validation_texts = [" ".join(sentence_array) for sentence_array in valid_x]

    if args.verbose:
        print(f'Raw Joined Data')
        print(f'*******')
        print(f'Array of Texts: {array_of_texts[0:2]}')
        print(f'*******')

    count_vect.fit(array_of_texts)

    if args.verbose:
        print(f'Count Vect (after fit)')
        print(f'{count_vect.get_feature_names()[0:10]}')
        print(f'*******')

    # transform the training and validation data using count vectorizer object
    xtrain_count =  count_vect.transform(array_of_training_texts)
    xvalid_count =  count_vect.transform(array_of_validation_texts)
    
    if args.verbose:
        print(f'Count Features')
        print(f'*******')
        print(f'Train X: {xtrain_count[0:2]}')
        print(f'Valid X: {xvalid_count[0:2]}')
        print(f'*******')

if args.feature == 'tfidf_word':
    # load the pre-trained word-embedding vectors 
    embeddings_index = {}
    for i, line in enumerate(open('data/wiki-news-300d-1M.vec')):
        values = line.split()
        embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')

    # create a tokenizer 
    token = text.Tokenizer()
    token.fit_on_texts(trainDF['text'])
    word_index = token.word_index

    # convert text to sequence of tokens and pad them to ensure equal length vectors 
    train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
    valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)

    # create token-embedding mapping
    embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector