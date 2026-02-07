import shutil

def remove_directory(out_data):
    """
    Removes the directory out_data (if one exists, otherwise does nothing)
    """
    if not out_data.exists():
        return
    shutil.rmtree(out_data)

def iter_images(data):
    """
    Helper function that helps us iterate through all images in the given data path. It yields
    every image file under the given directory (note all files should be images)
    """
    for d in data.rglob("*"):
        if d.is_file(): yield d

def gen_image_list(cat_loc):
    """
    Returns a list of the image file paths inside of the given directory
    """
    images = [file for file in cat_loc.iterdir() if file.isfile()]
    return sorted(images)