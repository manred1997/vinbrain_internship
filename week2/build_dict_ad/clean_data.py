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
                line = re.sub("[\(\)]", "", line)
                line = re.sub("[^\w\s]", "", line)
                # line = re.sub("\{\d\}", "x", line)
                if not line: continue
                else: samples.append(line)
            except: continue
    pprint.pprint(samples)
    with open("english_clean.txt", "w") as f:
        f.write("\n".join(samples))