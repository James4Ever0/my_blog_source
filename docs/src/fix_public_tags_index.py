filepath = "public/tags/index.html"

with open(filepath, "r") as f:
    content = f.read()
    lines = content.split("\n")
    lines = [it.strip() for it in lines if it.strip()]
with open(filepath, "w+") as f:
    for line in lines:
        f.write(line+"\n")