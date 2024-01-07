rm db.json
hexo generate
cd public
python3 -m http.server 8021