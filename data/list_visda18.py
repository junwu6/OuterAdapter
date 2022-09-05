import os
import random
p_path = os.path.join('E:\\Codes\\data\\visda-2018', 'train')
dir_list = os.listdir(p_path)
print(dir_list)

class_list = ["aeroplane", "bicycle", "bus", "car", "horse", "knife", "motorcycle", "person", "plant",
              "skateboard", "train", "truck", 'unk']
path_source = "visda-2018_Synthetic_source_list.txt"
write_source = open(path_source, "w")
for k, direc in enumerate(dir_list):
    if not '.txt' in direc:
        files = os.listdir(os.path.join(p_path, direc))
        for i, file in enumerate(files):
            if direc in class_list:
                class_name = direc
                file_name = os.path.join(p_path, direc, file)
                if random.random() < 0.05:
                    write_source.write('%s %s\n' % (file_name, class_list.index(class_name)))
            else:
                continue
p_path = os.path.join('E:\\Codes\\data\\visda-2018', 'validation')
dir_list = os.listdir(p_path)
print(dir_list)

for iii in range(1, 13):
    path_target = "visda-2018_Real_{}_target_list.txt".format(str(iii))
    write_target = open(path_target, "w")
    for k, direc in enumerate(dir_list):
        if not '.txt' in direc:
            files = os.listdir(os.path.join(p_path, direc))
            for i, file in enumerate(files):
                if direc in class_list:
                    class_name = direc
                else:
                    class_name = "unk"
                file_name = os.path.join(p_path, direc, file)
                if random.random() < 0.5:
                    if class_name == "unk":
                        if random.random() <= 0.01*iii:
                            write_target.write('%s %s\n' % (file_name, class_list.index(class_name)))
                    else:
                        write_target.write('%s %s\n' % (file_name, class_list.index(class_name)))
