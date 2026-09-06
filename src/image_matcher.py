import cv2
import numpy as np
# Dono test images load karo
image_a = cv2.imread("dataset/image_A.png", cv2.IMREAD_GRAYSCALE)
image_b = cv2.imread("dataset/image_B.png", cv2.IMREAD_GRAYSCALE)

# Check karo ki images successfully load hui hain
if image_a is None or image_b is None:
    print("Error: Could not load input images.")
    raise SystemExit

# SIFT feature detector
sift = cv2.SIFT_create()

# Features aur descriptors detect karo
keypoints_a, descriptors_a = sift.detectAndCompute(image_a, None)
keypoints_b, descriptors_b = sift.detectAndCompute(image_b, None)

print("Image A features:", len(keypoints_a))
print("Image B features:", len(keypoints_b))

# Check karo ki descriptors available hain
if descriptors_a is None or descriptors_b is None:
    print("Error: Could not detect usable features in one or both images.")
    raise SystemExit
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
# Homography ke liye kam se kam 4 matches zaroori hain
if len(good_matches) < 4:
    print("Not enough good matches for reliable registration.")
    raise SystemExit

# Homography ke liye kam se kam 4 matches zaroori hain
if len(good_matches) < 4:
    print("Not enough good matches for reliable registration.")
    raise SystemExit

# Good matches se point coordinates nikalo
src_pts = np.float32(
    [keypoints_a[m.queryIdx].pt for m in good_matches]
).reshape(-1, 1, 2)

dst_pts = np.float32(
    [keypoints_b[m.trainIdx].pt for m in good_matches]
).reshape(-1, 1, 2)

# RANSAC se Homography find karo
homography_matrix, mask = cv2.findHomography(
    src_pts,
    dst_pts,
    cv2.RANSAC,
    5.0
)

# Check karo ki Homography successfully calculate hui hai
if homography_matrix is None or mask is None:
    print("Error: Could not calculate a reliable homography.")
    raise SystemExit

# Valid matches (inliers) count karo
inliers = int(mask.sum())

print("Inliers after RANSAC:", inliers)
print("Inlier ratio:", inliers / len(good_matches)) 
# Image A ko Image B ke according align karo
height, width = image_b.shape

registered_image = cv2.warpPerspective(
    image_a,
    homography_matrix,
    (width, height)
)
# Sirf RANSAC inlier points select karo
inlier_mask = mask.ravel().astype(bool)
inlier_src_pts = src_pts[inlier_mask]
inlier_dst_pts = dst_pts[inlier_mask]

# Sirf reliable inlier points ko project karo
projected_inlier_pts = cv2.perspectiveTransform(
    inlier_src_pts,
    homography_matrix
)

# Inlier registration error calculate karo
inlier_errors = np.linalg.norm(
    projected_inlier_pts - inlier_dst_pts,
    axis=2
)

mean_error = float(np.mean(inlier_errors))

print("Mean registration error: {:.2f} pixels".format(mean_error))
# Registered image save karo
cv2.imwrite(
    "outputs/registered_image.png",
    registered_image
)

print("Registered image saved to outputs/registered_image.png") 
# Registration quality metrics
inlier_ratio_percent = (inliers / len(good_matches)) * 100

print("\n--- Registration Metrics ---")
print("Good matches:", len(good_matches))
print("RANSAC inliers:", inliers)
print("Inlier ratio: {:.2f}%".format(inlier_ratio_percent)) 
# Registered image aur Image B ko blend karo
overlay = cv2.addWeighted(
    registered_image,
    0.5,
    image_b,
    0.5,
    0
)

# Overlay result save karo
cv2.imwrite(
    "outputs/overlay.png",
    overlay
)

print("Overlay image saved to outputs/overlay.png")