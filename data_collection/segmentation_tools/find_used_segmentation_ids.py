import airsim
import numpy as np
from PIL import Image
from io import BytesIO

IMG_SIZE = (224, 224)

client = airsim.MultirotorClient()
client.confirmConnection()

responses = client.simGetImages([
    airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, True)
])
if responses and len(responses) > 0:
    img_bytes = responses[0].image_data_uint8
    if len(img_bytes) == 0:
        print("No segmentation image received.")
    else:
        mask = Image.open(BytesIO(img_bytes)).convert("L").resize(IMG_SIZE, Image.NEAREST)
        mask_np = np.array(mask)
        unique_ids = np.unique(mask_np)
        print("Unique segmentation IDs in current scene:", unique_ids)
        mask.save("segmentation_debug.png")
        print("Segmentation mask saved as segmentation_debug.png for inspection.")
else:
    print("No response from AirSim.")
