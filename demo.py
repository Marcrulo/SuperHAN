import sys
from PIL import Image

SUPERHAN_PATH = "/home/mp/Desktop/SuperHAN"
sys.path.insert(0, SUPERHAN_PATH)

from inference import SuperResolution, SuperHAN

img = Image.open("example.jpg")

# SR only
sr_model = SuperResolution(f"{SUPERHAN_PATH}/checkpoints/sr/best.pt")
sr_img = sr_model(img)
sr_img.show()

# SR + keypoints
model = SuperHAN(f"{SUPERHAN_PATH}/checkpoints/superfan/best.pt")
sr_img, keypoints = model(img)
sr_img.show()
print(keypoints)  # (21, 2) x,y coords in SR image space
