import sys
import string


in_file = sys.argv[1]

with open(in_file, "r", encoding="utf-8") as f:
    for line in f.readlines():
        line = line.strip().lower()
        for w in string.punctuation:
            line = line.replace(w, "")
        line = line.replace("  ", "")
        print(line)

