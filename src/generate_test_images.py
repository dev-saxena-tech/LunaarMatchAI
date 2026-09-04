import cv2
import numpy as np

# Canvas banate hain
image = np.zeros((600, 800), dtype=np.uint8)

# Kuch shapes/features draw karte hain
cv2.circle(image, (250, 250), 80, 255, 3)
cv2.circle(image, (550, 350), 60, 255, 3)
cv2.rectangle(image, (100, 400), (300, 550), 255, 3)
cv2.line(image, (450, 100), (700, 200), 255, 4)

# Image A save
cv2.imwrite("dataset/image_A.png", image)

# Image B = Image A ka thoda shifted version
matrix = np.float32([[1, 0, 40], [0, 1, 30]])
image_B = cv2.warpAffine(image, matrix, (800, 600))

# Image B save
cv2.imwrite("dataset/image_B.png", image_B)

print("Test images created successfully!")