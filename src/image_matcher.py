import cv2

# Dono test images load karo
image_a = cv2.imread("dataset/image_A.png", cv2.IMREAD_GRAYSCALE)
image_b = cv2.imread("dataset/image_B.png", cv2.IMREAD_GRAYSCALE)

# SIFT feature detector
sift = cv2.SIFT_create()

# Features aur descriptors detect karo
keypoints_a, descriptors_a = sift.detectAndCompute(image_a, None)
keypoints_b, descriptors_b = sift.detectAndCompute(image_b, None)

print("Image A features:", len(keypoints_a))
print("Image B features:", len(keypoints_b))
# Feature matcher
matcher = cv2.BFMatcher()

# Har feature ko doosri image ke features se compare karo
matches = matcher.knnMatch(
    descriptors_a,
    descriptors_b,
    k=2
)

# Lowe's Ratio Test se good matches select karo
good_matches = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print("Total matches:", len(matches))
print("Good matches:", len(good_matches)) 
# Good matches ko image par draw karo
matched_image = cv2.drawMatches(
    image_a,
    keypoints_a,
    image_b,
    keypoints_b,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Result save karo
cv2.imwrite("outputs/matched_features.png", matched_image)

print("Matched image saved to outputs/matched_features.png")