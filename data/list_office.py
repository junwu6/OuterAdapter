import os


def get_list(p_path, class_list, name, mode='target', chosen_unk=None):
    if 'home' not in name:
        p_path = os.path.join(p_path, 'images')
    dir_list = os.listdir(p_path)
    print(dir_list)

    write_file = open(name + '_' + mode + '_list.txt', "w")
    for k, direc in enumerate(dir_list):
        if not '.txt' in direc:
            files = os.listdir(os.path.join(p_path, direc))
            for i, file in enumerate(files):
                if direc in class_list:
                    class_name = direc
                    file_name = os.path.join(p_path, direc, file)
                    write_file.write('%s %s\n' % (file_name, class_list.index(class_name)))
                elif mode == 'target' and direc in chosen_unk:
                    class_name = "unk"
                    file_name = os.path.join(p_path, direc, file)
                    write_file.write('%s %s\n' % (file_name, class_list.index(class_name)))


if __name__ == '__main__':
    data_names = ['office-home']

    for data_name in data_names:
        print("Processing the data set: ", data_name)
        root, class_list, unk_list, domains = None, None, None, None
        if data_name == 'office':
            root = 'E:\Codes\data\office\domain_adaptation_images'
            class_list = ["back_pack", "bike", "bike_helmet", "bookcase", "bottle", "calculator", "desk_chair", "desk_lamp",
                          "desktop_computer", "file_cabinet", "unk"]
            domains = ['amazon', 'webcam', 'dslr']
        elif data_name == 'office-home':
            root = 'E:\Codes\data\office-home\images'
            class_list = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator',
                          'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp',
                          'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder',
                          'Fork', 'unk']
            unk_list = ['Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade', 'Laptop', 'Marker',
                        'Monitor', 'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip', 'Pen', 'Pencil',
                        'Postit_Notes', 'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 'Scissors',
                        'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 'Soda', 'Speaker', 'Spoon', 'Table', 'Telephone',
                        'ToothBrush', 'Toys', 'Trash_Can', 'TV', 'Webcam']
            domains = ['Art', 'Clipart', 'Product', 'Real_World']

        for src_domain in domains:
            for tgt_domain in domains:
                if tgt_domain == src_domain:
                    continue
                src_path = os.path.join(root, src_domain)
                src_name = data_name + '_' + src_domain
                get_list(src_path, class_list, src_name, mode='source')
                tgt_path = os.path.join(root, tgt_domain)

                for i in range(1, 11):
                    chosen_unk = unk_list[: 2*i]
                    tgt_name = data_name + '_' + tgt_domain + '_' + str(i)
                    get_list(tgt_path, class_list, tgt_name, mode='target', chosen_unk=chosen_unk)
