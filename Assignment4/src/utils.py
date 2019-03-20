def read_data(filename):
    data = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            data.append(list(map(int, line.split("	"))))
    f.close()
    return data