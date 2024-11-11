from PIL import Image
import auto.add_on.depth_pro as depth_pro
import matplotlib.pyplot as plt
import cv2

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()
image_path = "./images/4.jpg"

# Load and preprocess an image.
image, _, f_px = depth_pro.load_rgb(image_path)
image = transform(image)

# Run inference.
prediction = model.infer(image, f_px=f_px)
depth = prediction["depth"].numpy()  # Depth in [m].
focallength_px = prediction["focallength_px"]  # Focal length in pixels.

heatmapshow = None
heatmapshow = cv2.normalize(depth, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
cv2.imshow("Heatmap", heatmapshow)
cv2.waitKey(0)