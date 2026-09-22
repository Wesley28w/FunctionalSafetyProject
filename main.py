from ultralytics import YOLO

model = YOLO('yolo11n.pt')

# returns list of detections.
results = model(
    # provide a list of images
    [
        "assets/test.jpeg",
        "assets/wesley.png"
    ],
    # can filter classes. Find index for docs. 0 = Person
    classes=[0],
    save=True
    )

for result in results:
    print(result.boxes)