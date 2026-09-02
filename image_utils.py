from PIL import Image


def prepare_image(image_file):
    """
    Opens an uploaded image,
    displays its original size,
    resizes it,
    and returns the resized image.
    """

    # Open image
    image = Image.open(image_file)

    # Original size
    width, height = image.size

    print(f"Original Width : {width}")
    print(f"Original Height: {height}")

    # Resize image
    resized_image = image.resize((500, 500))

    return resized_image