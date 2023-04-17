
from random import randrange

hard_code_map = {
    "计算机": "软件工程师",
    "物流管理": "快递员",
    "市场营销": "销售",
    "临床医学": "医生"
}

train_data_path="./data/train.txt"
# origin text dim, 8 X 4
e = 8
f = 4

# result txt dim, 100 X 7
m = 100
n = 8

# label dim 
dim = 4

origin = [[0 for i in range(f)] for j in range(e)]
result = [[0 for i in range(n)] for j in range(m)]
# print(result)

def main():
    i = 0
    with open("./data/label-index.txt") as f:
        for line in f:
            j = 0
            for w in line.strip().split(","):
                origin[i][j] = w
                j += 1
            i += 1
    # print(origin)

    # origin[3] 与 origin[-1]强相关
    for p in range(m):
        for q in range(n):
            if q == n - 1:
                result[p][q] = hard_code_map[result[p][q-4]]
                break
            result[p][q] = origin[q][randrange(dim)]

    # print(result)
    with open(train_data_path, 'w') as t:
        for l in range(m):
            if l == m-1:
                t.write(",".join(result[l]))
                break
            t.write(",".join(result[l]) + "\n")

if __name__ == "__main__":
    main()