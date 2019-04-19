Following scripts are present in the project directory:
1. scraper.py - Used to scrape pcap files from MTA website
2. run_data_collection.py - Used to run scraper.py
3. pcap_to_text.py - Convert pcaps to text using tshark command -- Tshark needs to be installed on system
4. classify.py - Uses the text files to train a doc2vec model, trains svm and tests
5. script_runner.py - Used to run project - mainly pcap_to_text and classify

An example code which can put in script_runner:

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
