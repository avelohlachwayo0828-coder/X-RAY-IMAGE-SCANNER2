print("Step 1: Script started")

from PIL import Image
print("Step 2: PIL imported")

from gradcam import generate_gradcam
print("Step 3: gradcam imported")

image = Image.open(
    r"C:\Users\Avelo\Desktop\TB_Image_Scanner\tbx11k-simplified\test\unknown_1.png"
)
print("Step 4: Image loaded")

heatmap = generate_gradcam(image)
print("Step 5: Heatmap generated")

Image.fromarray(heatmap).save("gradcam_result.png")
print("Step 6: Image saved")

print("Finished successfully!")