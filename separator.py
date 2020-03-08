import os
import pprint
import shutil

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main():
    dataset_uri = os.path.join(os.getcwd(),"ham-and-spam-dataset")
    ham_set = os.path.join(dataset_uri,"ham")
    spam_set =  os.path.join(dataset_uri, "spam")

    new_datasets = os.path.join(os.getcwd(),"datasets")

    #create data_sets folder if not exists
    if not os.path.exists(new_datasets):
        os.makedirs(new_datasets)

    ham_file_names = os.listdir(ham_set)
    chunky_ham = chunks(ham_file_names,500)

    for num, chunk in enumerate(chunky_ham, start= 1):
        sub_directory_name = os.path.join(new_datasets,"ham"+str(num))

        if not os.path.exists(sub_directory_name):
            os.makedirs(sub_directory_name)

        sub_directory_file_names = [ os.path.join(ham_set,file_name) for file_name in chunk]
        for single_path in sub_directory_file_names:
            shutil.copy(single_path,sub_directory_name)

    spam_file_names = os.listdir(spam_set)
    chunky_spam = chunks(spam_file_names, 100)

    for num, chunk in enumerate(chunky_spam, start=1):
        sub_directory_name = os.path.join(new_datasets, "spam" + str(num))

        if not os.path.exists(sub_directory_name):
            os.makedirs(sub_directory_name)

        sub_directory_file_names = [os.path.join(spam_set, file_name) for file_name in chunk]
        for single_path in sub_directory_file_names:
            shutil.copy(single_path, sub_directory_name)



if __name__ == '__main__':
    main()