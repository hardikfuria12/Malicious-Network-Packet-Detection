import classify
import pcap_to_text
import os

TEXT_LIST_PICKLE_PATH = 'text_data/'
TEXT_LIST_PICKLE_PATH_2 = 'text_data_2/'

EXTRACTED_DATA_PATH = 'Extracted_data/'

VECTOR_LIST_PICKLE_PATH = 'vector_data/'  # Method_1
VECTOR_LIST_PICKLE_PATH_2 = 'vector_data_2/' # Method_2

if __name__ == '__main__':

    os.makedirs(TEXT_LIST_PICKLE_PATH, exist_ok=True)
    os.makedirs(TEXT_LIST_PICKLE_PATH_2, exist_ok=True)
    os.makedirs(EXTRACTED_DATA_PATH, exist_ok=True)
    os.makedirs(VECTOR_LIST_PICKLE_PATH, exist_ok=True)
    os.makedirs(VECTOR_LIST_PICKLE_PATH_2, exist_ok=True)

    list_of_malware_files = ['malware_2013', 'malware_2014', 'malware_2015', 'malware_D3M']
    list_of_normal_files = ['normal_Stratosphere_Normal', 'normal_Stratosphere_Normal_test']


    # # # Method 1

    # folder_paths = ['D3M/', 'MTA/2013/', 'MTA/2014/', 'MTA/2015/', 'MTA/2016/', 'MTA/2017/', 'MTA/2018/', 'Stratosphere_Malware/']
    # classify.dataloader('malware', folder_paths, EXTRACTED_DATA_PATH, TEXT_LIST_PICKLE_PATH, type=1)

    classify.train_svm(list_of_malware_files, list_of_normal_files, VECTOR_LIST_PICKLE_PATH, 'classifier6.model')

    list_of_vectors_to_test = ['normal_Stratosphere_Normal_test_2',
                               'malware_Stratosphere_Malware', 'malware_2016',
                               'malware_2017', 'malware_2018']

    list_of_classes = [1, 0, 0, 0, 0]

    classify.test_svm(list_of_vectors_to_test, list_of_classes, VECTOR_LIST_PICKLE_PATH, 'classifier6.model')


    # # # Method 2

    classify.train_svm(list_of_malware_files, list_of_normal_files, VECTOR_LIST_PICKLE_PATH_2,'classifier5.model')

    list_of_vectors_to_test = ['normal_Stratosphere_Normal_test_2',
                               'malware_Stratosphere_Malware', 'malware_2016',
                               'malware_2017', 'malware_2018']
    list_of_classes = [1, 0, 0, 0, 0]
    classify.test_svm(list_of_vectors_to_test, list_of_classes, VECTOR_LIST_PICKLE_PATH_2,'classifier5.model')


    ''' Example of project starting from scratch:

    Create a new folder in project's root and specify its name here
    If there are multiple datasets then create a folder and then inside it a folder per dataset which contains all the pcaps
    For example consider 'MTA/MTA_2014/abc.pcap' --> path_to_pcap = 'MTA/'

    path_to_pcap_folder = 'example_pcap_data/'
    # Specify the class of the pcap data. Create separate folders for different classes
    class_of_pcaps = 'malware'

    # Extract the text from pcap files and save to EXTRACTED_DATA_PATH
    folder_paths = pcap_to_text.main(path_to_pcap_folder, class_of_pcaps, EXTRACTED_DATA_PATH)

    # Load the data from EXTRACTED_DATA_PATH into a list with 100 lines as one item and save pickle to TEXT_LIST_PICKLE_PATH
    # type=1 is for method 1. Provide type = 2 for method 2
    classify.dataloader(class_of_pcaps, folder_paths, EXTRACTED_DATA_PATH, TEXT_LIST_PICKLE_PATH, type=1)

    # Specify the name of doc2vecmodel. Include .model at end as extension!
    doc2vec_model_name = 'doc2_vecz.model'
    #
    # # Trains doc2vecmodel from text_pickles listed in TEXT_LIST_PICKLE_PATH
    classify.train_doc2vec_model(TEXT_LIST_PICKLE_PATH, doc2vec_model_name)

    # List of training files
    # You will have to specify these manually for now!
    list_of_malware_files_for_training = ['malware_D3m', 'malware_example_pcap_data']
    list_of_normal_files_for_training = ['malware_strat']


    # Using the trained doc2vec model, first generate vectors and save them as a pickle for reuse.

    classify.generate_vector_pickles(doc2vec_model_name,TEXT_LIST_PICKLE_PATH,VECTOR_LIST_PICKLE_PATH, list_of_malware_files_for_training)
    classify.generate_vector_pickles(doc2vec_model_name,TEXT_LIST_PICKLE_PATH,VECTOR_LIST_PICKLE_PATH, list_of_normal_files_for_training)

    # SVM model name, it will be saved as 'svm_model3.pkl'
    svm_model_path = 'svm_model3'

    # Train svm model
    classify.train_svm(list_of_malware_files_for_training, list_of_normal_files_for_training, VECTOR_LIST_PICKLE_PATH, svm_model_path)

    # Test svm model
    # Need to provide this manually from VECTOR_LIST_PICKLE_PATH
    list_of_files_to_test = ['malware_D3m','malware_strat']
    classify.generate_vector_pickles(doc2vec_model_name,TEXT_LIST_PICKLE_PATH,VECTOR_LIST_PICKLE_PATH, list_of_files_to_test)

    # Need to provide this as 0 for malware_files and 1 for benign files!
    # Should be same length as list_of_files_to_test
    list_of_classes = [0, 1]

    classify.test_svm(list_of_files_to_test, list_of_classes, VECTOR_LIST_PICKLE_PATH, svm_model_path)
    '''