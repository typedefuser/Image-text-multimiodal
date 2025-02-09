import matplotlib.pyplot as plt
from PIL import Image

'''
visulaizing results
'''
def show_image_with_caption(image_path, caption):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis("off")
    plt.title(caption, fontsize=12)
    plt.show()
