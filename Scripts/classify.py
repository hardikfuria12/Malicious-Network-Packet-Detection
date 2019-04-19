from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from os import listdir, path
from sklearn.svm import SVC
import pickle as pkl
import pickle
import numpy
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import os

TEXT_LIST_PICKLE_PATH = 'text_data/'
TEXT_LIST_PICKLE_PATH_2 = 'text_data_2/'
EXTRACTED_DATA_PATH = 'Extracted_data_2/'

VECTOR_LIST_PICKLE_PATH = 'vector_data/'
VECTOR_LIST_PICKLE_PATH_2 = 'vector_data_2/'


def train_doc2vec_model(pickle_folder_path, model_name):

    data = []
    pickle_file_list = os.listdir(pickle_folder_path)

    for file in pickle_file_list:
        if '.pkl' in file:
            with open(pickle_folder_path + file, "rb") as input_file:
                data.extend(pkl.load(input_file))

    tagged_data = []
    for i, _d in enumerate(data):
        tagged_data.append(TaggedDocument(words=_d.split(), tags=[str(i)]))

    max_epochs = 30
    vec_size = 100
    alpha = 0.025

    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha,
                    min_alpha=0.025,
                    min_count=1,
                    dm=0)

    model.build_vocab(tagged_data)

    print("\n\nStarted training doc2vec model")

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    model.save(model_name)
    print("Model saved as:",model_name)



def train_svm(list_of_malware, list_of_normal, vector_pickle_path, classifier_path):
    data_malware = []
    data_normal = []

    list_of_pickles = os.listdir(vector_pickle_path)

    for file in list_of_pickles:
        if file.split('_vectors')[0] in list_of_malware:
            # print(file)
            with open(vector_pickle_path + file, "rb") as input_file:
                data = pkl.load(input_file)
            data_malware.extend(data)

        if file.split('_vectors')[0] in list_of_normal:
            # print(file)
            with open(vector_pickle_path + file, "rb") as input_file:
                data = pkl.load(input_file)
            data_normal.extend(data)

    print("\nMalware data length:",len(data_malware))
    print("Benign data length:",len(data_normal))


    data = []
    data.extend(data_malware)
    data.extend(data_normal)

    malware_labels = numpy.zeros(len(data_malware))
    normal_labels = numpy.ones(len(data_normal))

    labels = []
    labels.extend(malware_labels)
    labels.extend(normal_labels)

    mapping_with_label = {}

    for i, _ in enumerate(data):
        mapping_with_label[i] = labels[i]

    x_train_shuffle = [i for i, x in enumerate(data)]
    from random import shuffle
    shuffle(x_train_shuffle)

    x_train = []
    y_train = []
    for x in x_train_shuffle:
        x_train.append(data[x])
        y_train.append(mapping_with_label[x])


    print("Started SVM training")
    # svclassifier = SVC(kernel='rbf', gamma=0.4, C = 10, decision_function_shape='ovr', degree=3, verbose=True)

    svclassifier = SVC(kernel='rbf', gamma=0.1, C = 10, decision_function_shape='ovr', degree=3, verbose=True)
    svclassifier.fit(x_train, y_train)

    pickle_fname = classifier_path
    pkl.dump(svclassifier, open(pickle_fname, 'wb'))
    print("\nSVM model saved\n\n")


    # param_grid = {'C': [0.1, 1, 10, 20], 'gamma': [1, 0.1, 0.01, 0.001, 10]}
    #
    # # Make grid search classifier
    # clf_grid = GridSearchCV(SVC(), param_grid, verbose=1)
    #
    # # Train the classifier
    # clf_grid.fit(x_train, y_train)
    #
    # # clf = grid.best_estimator_()
    # print("Best Parameters:\n", clf_grid.best_params_)
    # print("Best Estimators:\n", clf_grid.best_estimator_)


def test_svm(file_list, class_list, vector_path, classifier_path):

    classifier = pkl.load(open(classifier_path, 'rb'))

    for i, file in enumerate(file_list):
        print("\n Testing file:", file)
        with open(vector_path + file+'_vectors.pkl', "rb") as input_file:
            test_data = pkl.load(input_file)

        y_test = [class_list[i] for x in test_data]

        print('Data size', len(test_data))
        print('Testing model')

        y_pred = classifier.predict(test_data)
        accuracy = (y_pred == y_test).sum()/len(y_pred)

        print('Accuracy:', accuracy*100, '%')

        print(classification_report(y_test,y_pred))


def generate_vector_pickles(doc2vec_model_file, pickle_open_path, pickle_save_path, file_list):

    model = Doc2Vec.load(doc2vec_model_file)

    for file in file_list:
        with open(pickle_open_path + file + '.pkl', "rb") as input_file:
            data = pkl.load(input_file)

        data_vectors = []

        print('Inferring vectors for file:', file)
        for i in data:
            data_vectors.append(model.infer_vector(i))

        new_file_name = file.split('.pkl')[0] + '_vectors.pkl'
        print('Saving vectors as:', new_file_name)

        with open(pickle_save_path + new_file_name, 'wb') as file_to_write:
            pickle.dump(data_vectors,file_to_write)


def dataloader(class_type, paths, extracted_data_path, text_pickle_path, type):
    list_of_files = []
    file_names = []
    for path in paths:
        files = os.listdir(extracted_data_path+path)
        for file in files:
            if class_type in file:
                list_of_files.append(extracted_data_path+path+'/'+file)
                file_names.append(path.split('/')[-1])

    print('\nLoading the list of files:', list_of_files)
    if type == 2:
        for j, file in enumerate(list_of_files):
            data = []
            with open(file, 'rb') as text_file:
                text_lines = text_file.readlines()
                i = 0
                while i + 100 <= len(text_lines):
                    para = ''
                    for line in text_lines[i:i + 100]:
                        para = para + str(line).strip()

                    data.append(para)
                    i = i + 50

                with open(TEXT_LIST_PICKLE_PATH_2 + class_type + '_' + file_names[j] + '.pkl', 'wb') as f:
                    pickle.dump(data, f)
    else:
        for j, file in enumerate(list_of_files):
            data = []
            with open(file, 'rb') as text_file:
                para = ''
                for i, line in enumerate(text_file.readlines()):
                    para = para + str(line).strip()
                    if i % 100 == 0 and i != 0:
                        data.append(para)
                        para = ''

                with open(text_pickle_path + class_type + '_' + file_names[j] + '.pkl', 'wb') as f:
                    pickle.dump(data, f)

    print('Data loading finished for class:',class_type)

