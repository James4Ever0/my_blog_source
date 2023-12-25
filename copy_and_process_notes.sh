rm -rf source/_posts/*.md
rclone sync /root/Desktop/works/notes_ssh_keys/notes notes --include "*.md"
# cp /root/Desktop/works/notes_ssh_keys/notes/*.md source/_posts
python3 remove_unwanted_notes.py