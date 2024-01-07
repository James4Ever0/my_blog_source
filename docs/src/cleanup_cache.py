import os
import tinydb
import sys

sys.path.append(
    "/media/root/Toshiba XG3/works/prometheous/document_agi_computer_control"
)

from cache_db_context import hash_file
cache_dir = "cache/"
source_dir= "notes"
db_path = "cache_db.json"

db = tinydb.TinyDB(db_path)
whitelist = []
for fname in os.listdir(source_dir):
    fpath = os.path.join(source_dir, fname)
    q = tinydb.Query().source.path == fpath
    it = db.get(q)
    if it is not None:
        white_target = it['target']['path']
        whitelist.append(white_target)
for fname in os.listdir(cache_dir):
    fpath = os.path.join(cache_dir, fname)
    if fpath not in whitelist:
        os.remove(fpath)