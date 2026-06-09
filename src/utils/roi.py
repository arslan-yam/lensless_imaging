from lensless_helpers.preprocessor import ALIGNMENT

TOP, LEFT = ALIGNMENT["top_left"]
HEIGHT = ALIGNMENT["height"]
WIDTH = ALIGNMENT["width"]


def crop_roi(image):
    return image[..., TOP : TOP + HEIGHT, LEFT : LEFT + WIDTH]
