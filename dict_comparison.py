

f_dict = {}
for i in range(10000):
    if i not in f_dict:
        f_dict[i] = {}
    for j in range(10000):
        if j not in f_dict[i]:
            f_dict[i][j] = {}
        for k in range(10000):
             if k not in f_dict[i][j]:
                f_dict[i][j][k] = [0]