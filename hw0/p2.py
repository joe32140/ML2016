import numpy as np
from PIL import Image
import sys

pic = Image.open(sys.argv[1])
pic = pic.rotate(180)
pic.save('ans2.png')

