def id_writer(read_path, write_path):
    ids =[]
    with open(read_path) as id_file:
        for line in id_file.readlines():
            line = line.split('/')
            ids.append(line[6].split('.jpg')[0])
    with open(write_path, 'w', encoding='utf-8') as output:
        for id in ids:
            output.write(id+'\n')

id_writer('Data/image_features/dm_feats/output', 'Data/DM_ID.txt')