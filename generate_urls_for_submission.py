base_url = "https://james4ever0.github.io/blog_quarto"
import os
for path in os.listdir("quarto_blog/myblog/posts"):
    print(f"{base_url}/posts/{path}/")