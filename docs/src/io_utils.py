
UTF8 = "utf-8"
def load_file(fname: str):
    with open(fname, "r", encoding=UTF8) as f:
        cnt = f.read()
    return cnt


def write_file(fname: str, content: str):
    with open(fname, "w+", encoding=UTF8) as f:
        f.write(content)
