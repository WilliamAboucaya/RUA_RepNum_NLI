from sklearn.metrics import cohen_kappa_score
from pprint import pprint
import csv
import sys

if __name__ == '__main__':
    oana_file = sys.argv[1]
    william_file = sys.argv[2]
    oana = []
    william = []
    annotations = []
    with open(oana_file, 'r', encoding='utf-8') as data:
        for line in csv.DictReader(data, delimiter=";"):
            #print(line["label"])
            if len(line["label"]):
                oana.append(line["label"])
                annotations.append(line)

    with open(william_file, 'r', encoding='utf-8') as data:
        for line in csv.DictReader(data, delimiter=','):
            #print(line["label"])
            william.append(line["label"])

    print(cohen_kappa_score(oana, william[0:len(oana)]))
    print(len(oana))
    for i in range(0, len(oana)):
        print(str(oana[i]) + str(william[i]))
        if oana[i] != william[i]:
            pprint(annotations[i])
            print("\n")
