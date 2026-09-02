from PIL import Image
from predict import predict_tb

image = Image.open("tbx11k-simplified/images/h0001.png")

result = predict_tb(image)

print()

print("Diagnosis:", result["diagnosis"])
print("Confidence:", result["confidence"], "%")
print("TB Probability:", result["tb_probability"], "%")
print("Normal Probability:", result["normal_probability"], "%")