import airsim

client = airsim.MultirotorClient()
client.confirmConnection()

# Replace with your actual object names from Unreal/AirSim
object_names = [
    "Tree", "Building", "Structure", "Ground", "Sky", "Car", "Person", "Bench", "Box", "Obstacle"
]

client.simSetSegmentationObjectID("Tree", 1)
client.simSetSegmentationObjectID("Building", 2)

print("Object Name -> Segmentation ID")
for name in object_names:
    try:
        obj_id = client.simGetSegmentationObjectID(name)
        print(f"{name}: {obj_id}")
    except Exception as e:
        print(f"{name}: Error ({e})")

print("\nAdd the IDs you want to keep to ALLOWED_IDS in your main script.")
