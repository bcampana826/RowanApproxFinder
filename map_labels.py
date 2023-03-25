import csv
import sys

# 1st command line arg is labels, print to stdout

with open(sys.argv[1], 'r') as file:
    label_num = 0
    labels = {}
    reader = csv.reader(file, delimiter=' ')

    for line in reader:
        tag = line[1]

        print("v " + str(line[0]), end = " ")

        if tag not in labels:
            labels[tag] = label_num
            print(label_num)
            label_num += 1
        else:
            print(labels[tag])

    file.close()
