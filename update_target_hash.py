import os
import tinydb
import sys

sys.path.append(
    "/media/root/Toshiba XG3/works/prometheous/document_agi_computer_control"
)

from cache_db_context import hash_file
cache_path = "cache/"
db_path = "cache_db.json"

db = tinydb.TinyDB(db_path)

for fname in os.listdir(cache_path):
    fpath = os.path.join(cache_path, fname)
    q = tinydb.Query().target.path == fpath
    it = db.get(q)
    if it is not None:
        new_it = it.copy()
        new_it['target']['hash'] = hash_file(fpath)
        db.upsert(new_it, cond = q)
