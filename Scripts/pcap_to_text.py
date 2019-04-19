'''
This script extracts information from each of the pcap files and creates a csv file.
'''
import os
# UNZIP_PATH = 'unzipped_pcaps/'
UNZIP_PATH = 'separated_pcaps/normal/'
OUTPUT_PATH = 'normal_pcap/extracted_txt/'
TXT_FILES_PATH = 'separated_pcaps/normal/text/'

def extract_from_pcap(class_type, path_to_pcap, path_to_text):
    if os.path.exists(path_to_pcap):
        list_of_files = os.listdir(path_to_pcap)
        list_of_files.sort()
        # print(list_of_files)
        # print(len(list_of_files))
        for i, file in enumerate(list_of_files):
            if '.pcap' in file:
                file_name = path_to_pcap+'/' + file
                print('Extracting information from file',i,':',file)
                # command = 'tshark -r '+ file_name +' -T fields -e _ws.col.Info > '+ TXT_FILES_PATH+'/'+class_type+str(i)+'.txt'
                # command = 'tshark -r' +file_name + ' -Y \"http||dns||smtp||ftp||pop||telnet\" -T fields -e _ws.col.Info > '+path_to_text+'/'+class_type+str(i)+'.txt'
                command = 'tshark -r' +file_name + ' -Y \"http||dns||smtp||ftp||pop||telnet\" > '+path_to_text+'/'+class_type+str(i)+'.txt'

                os.system(command)

def compile_in_one_text(class_type, text_data_path):
    list_of_files = os.listdir(text_data_path)
    # print(list_of_files)
    list_of_files.sort()
    with open(text_data_path+'/'+class_type+'.txt', 'wb') as f:
        for i, file in enumerate(list_of_files):
            if class_type in file and '.txt' in file:
                # print(file)
                with open(text_data_path+'/'+file, 'rb') as f1:
                    for line in f1:
                        # text = f1.read()
                        if 'Packet size limited during capture' not in str(line):
                            f.write(line)

def main(folder_path, class_type, extracted_data_path):

    list_of_folders = os.listdir(folder_path)

    list_of_folders_to_return = []

    for folder in list_of_folders:
        if os.path.isdir(os.path.join(folder_path, folder)) and folder != 'text_data':
            print('Extracting from folder:', folder)
            list_of_files = os.listdir(os.path.join(folder_path, folder))
            base_folder_path = os.path.join(folder_path, folder)
            # print(list_of_files)
            os.makedirs(base_folder_path+'/'+'text_data',exist_ok=True)
            text_data_path = os.path.join(base_folder_path,'text_data')
            extract_from_pcap(class_type, base_folder_path, text_data_path)
            compile_in_one_text(class_type, text_data_path)

            # Copy to extracted_data_path
            os.makedirs(extracted_data_path +'/' + folder, exist_ok=True)
            command_to_copy = 'cp ' + text_data_path + '/' + class_type + '.txt ' + extracted_data_path +'/' + folder + '/'
            os.system(command_to_copy)

            list_of_folders_to_return.append(folder)

        elif folder != 'text_data':
            print(folder)
            os.makedirs(folder_path + '/' + 'text_data', exist_ok=True)

            text_data_path = os.path.join(folder_path,'text_data')
            extract_from_pcap(class_type, folder_path, text_data_path)
            compile_in_one_text(class_type, text_data_path)

            # Copy to extracted_data_path
            # new_folder_name = folder.split('')
            os.makedirs(extracted_data_path + '/' + folder_path, exist_ok=True)
            command_to_copy = 'cp ' + text_data_path + '/' + class_type + '.txt ' + extracted_data_path + '/' + folder_path + '/'
            os.system(command_to_copy)

            list_of_folders_to_return.append(folder_path[:-1])

    return list(set(list_of_folders_to_return))

# if __name__ == '__main__':
#     # extract_from_pcap('malware')
#     # extract_from_pcap('normal')
#     # compile_in_one_text('malware')
#     # compile_in_one_text('normal')
#
#     list_of_folders = main('pcap_data/','malware', 'Extracted_data')
#     print(list_of_folders)