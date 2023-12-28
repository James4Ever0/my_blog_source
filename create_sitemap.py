template_path = "sitemap_template.xml.j2"
# template_path = "sitemap_template.html.j2"
output_file_path = "sitemap.xml"
# output_file_path = "sitemap.html"
urls_path = "urls.txt"
base_url = "https://james4ever0.github.io"
from jinja2 import Environment, FileSystemLoader

# Load the template from file
file_loader = FileSystemLoader(
    "."
)  # Replace 'path_to_templates_directory' with the actual path
env = Environment(loader=file_loader)
template = env.get_template(
    template_path
)  # Replace 'sitemap_template.html' with the actual template file name

with open(urls_path, "r") as f:
    content = f.read()
    lines = content.split("\n")
    lines = [it.strip() for it in lines if it.strip()]

# Data to be rendered
datalist = [(item, "2023-12-28T09:21:02+00:00", "1.00") for item in lines]
# datalist = [(str(index), item.replace(base_url, "")) for index, item in enumerate(lines)]

# Render the template with the data
rendered_template = template.render(datalist=datalist)

# Write the rendered output to a file
with open(output_file_path, "w") as output_file:
    output_file.write(rendered_template)

print("Template rendered and written to file successfully.")
