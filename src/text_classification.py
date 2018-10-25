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
parser.add_argument('--feature', choices=['count', 'tf_idf_word', 'tf_idf_ngram', 'tf_idf_char'],
                    default='count', nargs='+')
parser.add_argument('--classifier', choices=['naive_bayes'],
                    default='naive_bayes')
parser.add_argument('--newdata', action='store_true',
                   help='tests model on new data')

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

array_of_texts = [" ".join(sentence_array) for sentence_array in trainDF.text.values]
array_of_training_texts = [" ".join(sentence_array) for sentence_array in train_x]
array_of_validation_texts = [" ".join(sentence_array) for sentence_array in valid_x]

array_of_new_data = []
if args.newdata:
    new_data = open('data/new_data.txt').read()
    for i, line in enumerate(new_data.split('\n')):
        array_of_new_data.append(line)

if args.verbose:
    print(f'Raw Joined Data')
    print(f'*******')
    print(f'Array of Texts: {array_of_texts[0:2]}')
    print(f'*******')


if 'count' in args.feature:
    # create a count vectorizer object 
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    
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
    
    if args.newdata:
        xnewdata_vector = count_vect.transform(array_of_new_data)

if 'tf_idf_word' in args.feature:
    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(array_of_texts)

    if args.verbose:
        print(f'TF-IDF Vect (after fit)')
        print(f'{tfidf_vect.get_feature_names()[0:10]}')
        print(f'*******')

    xtrain_tfidf =  tfidf_vect.transform(array_of_training_texts)
    xvalid_tfidf =  tfidf_vect.transform(array_of_validation_texts)
    
    if args.verbose:
        print(f'TF-IDF Features')
        print(f'*******')
        print(f'Train X: {xtrain_tfidf[0:2]}')
        print(f'Valid X: {xvalid_tfidf[0:2]}')
        print(f'*******')
    
    if args.newdata:
        xnewdata_vector = tfidf_vect.transform(array_of_new_data)

if 'tf_idf_ngram' in args.feature:
    # ngram level tf-idf 
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram.fit(array_of_texts)

    if args.verbose:
        print(f'TF-IDF Vect (after fit)')
        print(f'{tfidf_vect_ngram.get_feature_names()[0:10]}')
        print(f'*******')

    xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(array_of_training_texts)
    xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(array_of_validation_texts)

    if args.verbose:
        print(f'TF-IDF Features')
        print(f'*******')
        print(f'Train X: {xtrain_tfidf_ngram[0:2]}')
        print(f'Valid X: {xvalid_tfidf_ngram[0:2]}')
        print(f'*******')
    
    if args.newdata:
        xnewdata_vector = tfidf_vect_ngram.transform(array_of_new_data)

if 'tf_idf_char' in args.feature:
    # characters level tf-idf
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram_chars.fit(array_of_texts)
    if args.verbose:
        print(f'TF-IDF Vect (after fit)')
        print(f'{tfidf_vect_ngram_chars.get_feature_names()[0:10]}')
        print(f'*******')

    xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(array_of_training_texts) 
    xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(array_of_validation_texts)

    if args.verbose:
        print(f'TF-IDF Features')
        print(f'*******')
        print(f'Train X: {xtrain_tfidf_ngram_chars[0:2]}')
        print(f'Valid X: {xvalid_tfidf_ngram_chars[0:2]}')
        print(f'*******')
    
    if args.newdata:
        xnewdata_vector = tfidf_vect_ngram_chars.transform(array_of_new_data)

def train_model(classifier, feature_vector_train, label, feature_vector_valid, new_data_vector=None, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    if args.newdata:
        new_data_predictions = classifier.predict(new_data_vector)
        if args.verbose:
            print(f'New Data Predictions: {new_data_predictions[60:80]}')
            print(f'Raw New Data:')
            print("\n*".join(array_of_new_data[60:80]))
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)

if args.classifier == 'naive_bayes':
    new_data_vector = None
    if args.newdata:
        new_data_vector = xnewdata_vector
    if 'count' in args.feature:
        # Naive Bayes on Count Vectors
        accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count, new_data_vector)
        print(f'NB, Count Vectors: {accuracy}')

    if 'tf_idf_word' in args.feature:
        # Naive Bayes on Word Level TF IDF Vectors
        accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf, new_data_vector)
        print(f'NB, WordLevel TF-IDF: {accuracy}')

    if 'tf_idf_ngram' in args.feature:
        # Naive Bayes on Ngram Level TF IDF Vectors
        accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, new_data_vector)
        print(f'NB, N-Gram Vectors: {accuracy}')

    if 'tf_idf_char' in args.feature:
        # Naive Bayes on Character Level TF IDF Vectors
        accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars, new_data_vector)
        print(f'NB, CharLevel Vectors: {accuracy}')
