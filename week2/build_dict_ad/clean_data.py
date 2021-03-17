import re
import pprint

if __name__ == "__main__":
    # pp = pprint.PrettyPrinter(indent=4)
    samples = []
    with open("english.txt", "r") as f:
        for line in f.readlines():
            try:
                line = line.split("=")[1]
                line = re.sub("\\n", "", line)
                if not line: continue
                else: samples.append(line)
            except: continue
    pprint.pprint(samples)