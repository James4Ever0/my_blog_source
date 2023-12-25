rm -rf source/_posts/*.md
rclone sync /root/Desktop/works/notes_ssh_keys/notes notes --include "*.md"
# cp /root/Desktop/works/notes_ssh_keys/notes/*.md source/_posts

# before that, run:
# bash "/media/root/Toshiba XG3/works/prometheous/document_agi_computer_control/setup_openai_local_service.sh"

export TOKENIZERS_PARALLELISM=false

export OPENAI_API_KEY='any'
export OPENAI_API_BASE=http://0.0.0.0:8000
export BETTER_EXCEPTIONS=1

python3 remove_unwanted_notes.py