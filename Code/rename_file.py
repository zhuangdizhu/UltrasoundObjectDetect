import os

count = 0
label_name = 'static_inbox_empty'
file_path = "../Data/inbox/static/empty/"
for file_name in os.listdir(file_path):
    new_name = label_name + str(count) + '.wav'
    count +=1
    os.rename(file_path + file_name, file_path + new_name)
