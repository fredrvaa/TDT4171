def read_data(filename):
    data = []
    file = open(filename, 'r')
    lines = file.readlines()
    for line in lines:
        data.append(list(map(int,line.split())))
    file.close()
    return data