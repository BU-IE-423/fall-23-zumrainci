import numpy as np
import cv2
from scipy.stats import norm
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import random


#METHOD1
image_path = '0005.jpg'  
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

hist_values, bin_edges = np.histogram(image.flatten(), bins=256, range=(0, 255))

mu, std = norm.fit(image.flatten())

#histogram
plt.figure(figsize=(10, 5))
plt.bar(bin_edges[:-1], hist_values, width=1, edgecolor='black')
plt.title('Histogram of Pixel Values with Normal Distribution Fit')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p * max(hist_values), 'k', linewidth=2)
plt.show()

#estimated mean and standard deviation
print(f"Estimated Mean (mu): {mu:.2f}")
print(f"Estimated Standard Deviation (std): {std:.2f}")

#the bounds for 0.001 probability
lower_bound = norm.ppf(0.001, mu, std)
upper_bound = norm.ppf(0.999, mu, std)
print(f"Lower Bound (0.001 probability): {lower_bound:.2f}")
print(f"Upper Bound (0.999 probability): {upper_bound:.2f}")

outliers_global = np.logical_or(image < lower_bound, image > upper_bound)

#outlier pixels to zero (black)
modified_image_global = np.copy(image)
modified_image_global[outliers_global] = 0

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

#original image
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

#modified image
axes[1].imshow(modified_image_global, cmap='gray')
axes[1].set_title('Modified Image with Global Outliers Set to Black')
for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()

#PATCH-BASED ANALYSIS
def analyze_patches(image, patch_size=51):
    height, width = image.shape
    modified_image = np.copy(image)
    
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            # Make sure the patch is within the image bounds
            if (i + patch_size <= height) and (j + patch_size <= width):
                # Extract the patch
                patch = image[i:i+patch_size, j:j+patch_size].flatten()
                
                # Fit the normal distribution to the patch
                patch_mu, patch_std = norm.fit(patch)
                
                # Calculate the bounds for 0.05 probability
                lower_bound = norm.ppf(0.001, patch_mu, patch_std)
                upper_bound = norm.ppf(1-0.001, patch_mu, patch_std)
                
                # Determine the outliers
                outliers = np.where((patch < lower_bound) | (patch > upper_bound))
                
                # Set the outlier pixels to black
                patch[outliers] = 0
                
                # Reshape and insert the modified patch back into the image
                modified_patch = patch.reshape((patch_size, patch_size))
                modified_image[i:i+patch_size, j:j+patch_size] = modified_patch
    
    return modified_image

modified_image_patches = analyze_patches(image)

#the original and the modified image
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(modified_image_patches, cmap='gray')
ax[1].set_title('Patches Analysis Result')
for a in ax:
    a.axis('off')
plt.tight_layout()
plt.show()

#METHOD 2

def control_chart_analysis(image, axis=0):
    size = image.shape[axis]
    modified_image = np.copy(image)

    #control limits for mean and variance (3-sigma)
    pixel_means = np.mean(image, axis=axis)
    pixel_vars = np.var(image, axis=axis)
    mean_control_limit = [np.mean(pixel_means) - 3*np.std(pixel_means), np.mean(pixel_means) + 3*np.std(pixel_means)]
    var_control_limit = [np.mean(pixel_vars) - 3*np.std(pixel_vars), np.mean(pixel_vars) + 3*np.std(pixel_vars)]

    for i in range(size):
        if axis == 0:  # Rows
            row_mean = np.mean(image[i, :])
            row_var = np.var(image[i, :])
            if row_mean < mean_control_limit[0] or row_mean > mean_control_limit[1] or \
               row_var < var_control_limit[0] or row_var > var_control_limit[1]:
                modified_image[i, :] = 0
        else:  # Columns
            col_mean = np.mean(image[:, i])
            col_var = np.var(image[:, i])
            if col_mean < mean_control_limit[0] or col_mean > mean_control_limit[1] or \
               col_var < var_control_limit[0] or col_var > var_control_limit[1]:
                modified_image[:, i] = 0

    return modified_image, mean_control_limit, var_control_limit

#perform control chart analysis for rows and columns
modified_image_rows, mean_limit_rows, var_limit_rows = control_chart_analysis(image, axis=0)
modified_image_cols, mean_limit_cols, var_limit_cols = control_chart_analysis(image, axis=1)

#plot for row analysis
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(modified_image_rows, cmap='gray')
ax[1].set_title('Row Control Chart Analysis')
for a in ax:
    a.axis('off')
plt.tight_layout()
plt.show()

#plot for column analysis
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(modified_image_cols, cmap='gray')
ax[1].set_title('Column Control Chart Analysis')
for a in ax:
    a.axis('off')
plt.tight_layout()
plt.show()

#the control limits for rows and columns
print("Row Mean Control Limits:", mean_limit_rows)
print("Row Variance Control Limits:", var_limit_rows)
print("Column Mean Control Limits:", mean_limit_cols)
print("Column Variance Control Limits:", var_limit_cols)

#OUR PROPOSAL
#LBP (LOCAL BINARY PATTERN)

def calculate_lbp_histogram(image, P=16, R=2):
    lbp_image = local_binary_pattern(image, P, R, method='uniform')
    lbp_histogram, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, P*2+3), range=(0, P*2+2), density=True)
    return lbp_histogram

def detect_defects_lbp(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    lbp_hist = calculate_lbp_histogram(img)
    
    mean_hist = np.mean(lbp_hist)
    std_hist = np.std(lbp_hist)
    
    UCL = mean_hist + 3 * std_hist
    LCL = mean_hist - 3 * std_hist
    
    defect_regions = (lbp_hist > UCL) | (lbp_hist < LCL)
    
    return np.where(defect_regions)[0]

def modify_image_with_lbp_control(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    lbp = local_binary_pattern(img, P=16, R=2, method='uniform')

    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)

    mean_hist = np.mean(lbp_hist)
    std_hist = np.std(lbp_hist)

    UCL = mean_hist + 3 * std_hist
    LCL = mean_hist - 3 * std_hist

    out_of_control_lbp = np.where((lbp_hist > UCL) | (lbp_hist < LCL))[0]

    mask = np.isin(lbp, out_of_control_lbp)

    modified_img = img.copy()
    modified_img[mask] = 0

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(modified_img, cmap='gray')
    plt.title('Modified Image with Defects')

    plt.show()

    return modified_img

for i in range(10):
    x = random.randint(2,196)
    print(x)

image_path1 = '0005.jpg'
defects = detect_defects_lbp(image_path1)
print('Defective LBP regions:', defects)
modified_image = modify_image_with_lbp_control(image_path1)

image_path2 = '0004.jpg'
defects = detect_defects_lbp(image_path2)
print('Defective LBP regions:', defects)
modified_image = modify_image_with_lbp_control(image_path2)

image_path3 = '0031.jpg'
defects = detect_defects_lbp(image_path3)
print('Defective LBP regions:', defects)
modified_image = modify_image_with_lbp_control(image_path3)

image_path4 = '0034.jpg'
defects = detect_defects_lbp(image_path4)
print('Defective LBP regions:', defects)
modified_image = modify_image_with_lbp_control(image_path4)

image_path5 = '0135.jpg'
defects = detect_defects_lbp(image_path5)
print('Defective LBP regions:', defects)
modified_image = modify_image_with_lbp_control(image_path5)

image_path6 = '0141.jpg'
defects = detect_defects_lbp(image_path6)
print('Defective LBP regions:', defects)
modified_image = modify_image_with_lbp_control(image_path6)