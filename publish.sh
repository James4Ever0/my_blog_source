python3 fix_public_tags_index.py

# TODO: detect & remove headline from the markdown post.

cp -R public/* /media/root/Prima/hexo_blog_demo/publish/blog
cp -R quarto_blog/myblog/_site/* /media/root/Prima/hexo_blog_demo/publish/blog_quarto

cd /media/root/Prima/hexo_blog_demo/publish/blog
git add .
git commit -m "update blog"
git push -u origin main

cd /media/root/Prima/hexo_blog_demo/publish/blog_quarto
git add .
git commit -m "update blog"
git push -u origin main