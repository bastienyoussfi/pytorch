from fastbook import *

def search_images(term, max_images):
    print(f"Searching for '{term}'")
    urls = search_images_ddg(term, max_images)
    for result in urls:
        # Process each search result
        print(result)
    return urls