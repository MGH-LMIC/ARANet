from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.exposure import match_histograms
import seg_utils
import torch
import os
from natsort import natsorted
from tqdm import tqdm
import pandas as pd
import ast
import glob
from scipy.ndimage import gaussian_filter
import nibabel as nib

from skimage import measure
from stl import mesh
import os
import joblib


def nifti_to_stl(input_path, output_path, threshold=0.5, smoothing=1):
    try:
        # Load the NIFTI file
        nifti_img = nib.load(input_path)
        nifti_data = nifti_img.get_fdata()

        # Normalize data to 0-1 range if needed
        if nifti_data.max() > 1:
            nifti_data = nifti_data / nifti_data.max()

        # Extract the isosurface using marching cubes
        verts, faces, normals, values = measure.marching_cubes(
            nifti_data, threshold, step_size=smoothing, allow_degenerate=False
        )

        # Create the mesh
        stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

        # Transfer vertices from marching cubes output to STL mesh
        for i, face in enumerate(faces):
            for j in range(3):
                stl_mesh.vectors[i][j] = verts[face[j]]

        # Save the mesh to STL file
        stl_mesh.save(output_path)

        print(f"Successfully converted {input_path} to {output_path}")
        print(f"Mesh contains {len(faces)} faces and {len(verts)} vertices")
        return True

    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False


def read_case_files(path):
    files = natsorted(glob.glob(f"{path}/*.png"))
    return files


def create_masked_niftis(images, masks, output_dir, sigma=1):
    os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(png_output_dir, exist_ok=True)

    fluid_images = []
    fetus_images = []

    for i, (img, mask) in enumerate(zip(images, masks)):
        if isinstance(img, str):
            img = Image.open(img).convert("L")  # Convert to grayscale
        elif not isinstance(img, Image.Image):
            raise ValueError(
                "Each image must be either a file path or a PIL Image object"
            )

        if isinstance(mask, str):
            mask = Image.open(mask).convert("L")
        elif not isinstance(mask, Image.Image):
            raise ValueError(
                "Each mask must be either a file path or a PIL Image object"
            )

        img_array = np.array(img)
        mask_array = np.array(mask)

        if img_array.shape != mask_array.shape:
            mask = mask.resize(img_array.shape)
            mask_array = np.array(mask)

        # Create masked images for fluid (value 1) and fetus (value 2)
        fluid_mask = (mask_array == 1).astype(np.float32)
        fetus_mask = (mask_array == 2).astype(np.float32)

        # Apply Gaussian smoothing to the masks
        fluid_mask_smooth = gaussian_filter(fluid_mask, sigma=sigma)
        fetus_mask_smooth = gaussian_filter(fetus_mask, sigma=sigma)

        # Apply smoothed masks
        fluid_img = (img_array * fluid_mask_smooth).astype(np.uint8)
        fetus_img = (img_array * fetus_mask_smooth).astype(np.uint8)

        # # Save the masked images as PNG
        # Image.fromarray(fluid_img).save(os.path.join(png_output_dir, f'fluid_image_{i}.png'))
        # Image.fromarray(fetus_img).save(os.path.join(png_output_dir, f'fetus_image_{i}.png'))

        fluid_images.append(fluid_img)
        fetus_images.append(fetus_img)

    # Create and save NIfTI for fluid
    print(len(fluid_images))
    fluid_array = np.stack(fluid_images, axis=2).astype(np.float32)
    fluid_nifti = nib.Nifti1Image(fluid_array, affine=np.eye(4))
    fluid_nifti.header.set_xyzt_units(2)
    fluid_nifti.header["pixdim"][1:4] = [1.0, 1.0, 1.0]
    nib.save(fluid_nifti, os.path.join(output_dir, "fluid.nii"))

    # Create and save NIfTI for fetus
    fetus_array = np.stack(fetus_images, axis=2).astype(np.float32)
    fetus_nifti = nib.Nifti1Image(fetus_array, affine=np.eye(4))
    fetus_nifti.header.set_xyzt_units(2)
    fetus_nifti.header["pixdim"][1:4] = [1.0, 1.0, 1.0]
    nib.save(fetus_nifti, os.path.join(output_dir, "fetus.nii"))

    # return fluid_nifti, fetus_nifti


def separate_masks(original_mask):
    # Create mask A and mask B by copying the original mask
    mask_A = original_mask.copy()
    mask_B = original_mask.copy()

    # In mask A, convert values 2 to 1
    mask_A[mask_A == 2] = 0

    # In mask B, keep only values 0 and 2, then convert values 2 to 1
    mask_B[mask_B == 1] = 0
    mask_B[mask_B == 2] = 1

    return mask_A, mask_B


def display_masks(mask_A, mask_B):
    # Display mask A
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(mask_A, cmap="gray")
    plt.title("AF mask")
    plt.axis("off")

    # Display mask B
    plt.subplot(1, 2, 2)
    plt.imshow(mask_B, cmap="gray")
    plt.title("fetal mask")
    plt.axis("off")

    plt.show()


def get_bbox(mask):
    if np.any((mask == 1)):
        # Get the coordinates of all white (also shades of whites)
        # pixels in the numpy array
        grey_pixels = np.where(mask == 1)

        # Get the bounding box of the white pixels
        min_x = np.min(grey_pixels[1])
        min_y = np.min(grey_pixels[0])
        max_x = np.max(grey_pixels[1])
        max_y = np.max(grey_pixels[0])

        bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
    else:
        bbox = [0, 0, 0, 0]
    return bbox


def intersection(b1, b2):
    """
    Calculate the intersection of two bounding boxes.

    Parameters:
    b1, b2 (tuple or list): Bounding boxes in the format (x, y, w, h).

    Returns:
    tuple: Intersection bounding box in the format (x, y, w, h), or None if there is no intersection.
    """
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    # Calculate the coordinates of the intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    # Check if there is an intersection
    if x_right < x_left or y_bottom < y_top:
        return [0, 0, 0, 0]

    return [x_left, y_top, x_right - x_left, y_bottom - y_top]


def apply_bbox_to_mask(mask, bbox):
    """
    Apply a bounding box to a mask, converting values outside the box to 0.

    Parameters:
    mask (np.array): The mask to apply the bounding box to.
    bbox (tuple or list): The bounding box in the format (x, y, w, h).

    Returns:
    np.array: The mask with the bounding box applied.
    """
    # Create a new mask of zeros with the same shape as the input mask
    new_mask = np.zeros_like(mask)

    # Get the bounding box coordinates
    x, y, w, h = bbox

    # Apply the bounding box to the mask
    new_mask[y : y + h, x : x + w] = mask[y : y + h, x : x + w]

    return new_mask


def display_mask(mask):
    """
    Display a mask using matplotlib.

    Parameters:
    mask (np.array): The mask to display.
    """
    plt.imshow(mask, cmap="gray")
    plt.axis("off")
    plt.show()


def intersection_area(mask1, mask2):
    """
    Calculate the area of intersection between two masks.

    Parameters:
    mask1 (np.array): First binary mask.
    mask2 (np.array): Second binary mask.

    Returns:
    int: Area of intersection.
    """
    # Ensure the masks are binary
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    # Calculate intersection
    intersection = np.logical_and(mask1, mask2)

    # Calculate area of intersection
    area = np.sum(intersection)

    return area


def calculate_white_area(mask):
    # Count the number of white pixels (255)
    white_area = np.sum(mask == 1)
    return white_area


def resize_image(im, dim):
    im = Image.fromarray(im)
    dims = [im.size[0], im.size[1]]
    factor = dim / max(dims)
    resized_im = im.resize((round(im.size[0] * factor), round(im.size[1] * factor)))

    # Setting the points for cropped image
    left = -1 * ((dim - resized_im.size[0]) / 2)
    top = -1 * ((dim - resized_im.size[1]) / 2)
    right = (dim + resized_im.size[0]) / 2
    bottom = (dim + resized_im.size[1]) / 2
    resized = np.array(resized_im.crop((left, top, right, bottom)))
    # print(dims)
    return np.array(resized)


def upscale_image(im, dims, dim=256):
    im = Image.fromarray(im)
    factor = (max(dims)) / dim
    resized_im = im.resize((round(im.size[0] * factor), round(im.size[1] * factor)))
    return np.array(resized_im)


#######################################################################################


def process_files(df, idx):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = torch.load(
        "<class 'aranetfpn_aspp2.ARANetFPN'>_256_split_0.pt", map_location=DEVICE
    ).to(
        DEVICE
    )  # load ARANet segmenation model
    model = model.eval()
    clf = joblib.load("MRI_svc_model.joblib")  # load SVC classifier
    sclr = joblib.load("MRI_scaler.joblib")  # load SVC scaler

    fetal_volumes = {}
    AF_volumes = []
    sac_volumes = []
    fafo_cases = []

    rng = list(range(len(df)))
    # for k in tqdm(rng):
    k = idx
    voxel = df.loc[k, "voxels"]
    path = df.loc[k, "path"]
    age = df.loc[k, "age"]
    case = path.split("/")[-1]
    files = natsorted(os.listdir(path))
    images = []
    for i in files:
        s = os.path.join(path, i)
        images.append(s)

    all_preds = []

    for c in range(0, len(images)):
        preds = seg_utils.check_accuracy(images[c], model)
        all_preds.append(preds)

    print(len(all_preds))
    fetal_vol = 0
    AF_vol = 0
    sac_vol = 0
    fafo = 0
    for i in range(len(all_preds)):
        original_mask = all_preds[i]
        original_mask = upscale_image(
            original_mask, ast.literal_eval(df.loc[k, "dims"])
        )
        AF_mask, fetal_mask = separate_masks(original_mask)

        fetal_vol += calculate_white_area(fetal_mask)
        amniotic_sac_mask = AF_mask + fetal_mask
        sac_vol += calculate_white_area(amniotic_sac_mask)
        # display_masks(AF_mask, fetal_mask)
        af_bbox = get_bbox(AF_mask)
        fetal_bbox = get_bbox(fetal_mask)
        print(af_bbox, fetal_bbox)
        I_bbox = intersection(af_bbox, fetal_bbox)
        print(I_bbox)
        AF_mask_I = apply_bbox_to_mask(AF_mask, I_bbox)
        # display_mask(AF_mask_I)
        fetal_mask_I = apply_bbox_to_mask(fetal_mask, I_bbox)
        # display_mask(fetal_mask_I)
        fetal_mask_I_plus = AF_mask_I + fetal_mask_I
        # display_mask(fetal_mask_I_plus)
        fafo += intersection_area(AF_mask_I, fetal_mask_I_plus)

        # fetal_volumes[case] = round(fetal_vol,2)
        # sac_volumes.append(round(sac_vol,2))
        # fafo_cases.append(round(fafo,2))

    oligo_masks = []
    if (fafo / (sac_vol + 1e-8)) < 0.06:  # (fafo/sac_vol)
        for x in range(len(all_preds)):
            original_mask = all_preds[x]
            original_mask = upscale_image(
                original_mask, ast.literal_eval(df.loc[k, "dims"])
            )
            AF_mask, fetal_mask = separate_masks(original_mask)
            af_bbox = get_bbox(AF_mask)
            fetal_bbox = get_bbox(fetal_mask)
            # print(af_bbox, fetal_bbox)
            I_bbox = intersection(af_bbox, fetal_bbox)
            # print(I_bbox)
            AF_mask_I = apply_bbox_to_mask(AF_mask, I_bbox)
            combined_mask = AF_mask_I + (fetal_mask * 2)
            oligo_masks.append(combined_mask)
            AF_vol += calculate_white_area(AF_mask_I)
        oligoArray = np.stack(oligo_masks)
        # np.save(f'{case}_oligo_masks.npy',oligoArray)

    else:
        for z in range(len(all_preds)):
            original_mask = all_preds[z]
            original_mask = upscale_image(
                original_mask, ast.literal_eval(df.loc[k, "dims"])
            )
            AF_mask, fetal_mask = separate_masks(original_mask)
            AF_vol += calculate_white_area(AF_mask)

    AF_vol = (AF_vol * voxel) / 1000
    fetal_vol = (fetal_vol * voxel) / 1000
    sac_vol = (sac_vol * voxel) / 1000
    fafo = (fafo * voxel) / 1000

    print(fafo, AF_vol, age, fetal_vol)

    fafo_log = np.log((fafo * AF_vol * age) / fetal_vol + 600)  # claculate FAFO3D

    diagnosis = None

    output_masks = all_preds

    # classify case

    array = np.array([fafo_log, AF_vol]).reshape(-1, 2)
    scaled = sclr.transform(array)
    prediction = clf.predict(scaled)

    if prediction == 1:
        diagnosis = "Oligohydramnios"
    elif prediction == 0:
        diagnosis = "Normal"
    elif prediction == 2:
        diagnosis = "Polyhydramnios"

    return AF_vol, fafo_log, fetal_vol, diagnosis, output_masks, images


def colorize_images(images, ref_path, outdir):
    reference = np.array(Image.open(ref_path).convert("RGB"))
    for i in range(len(images)):
        image = np.array(Image.open(images[i]).convert("RGB"))
        matched = match_histograms(image, reference, channel_axis=-1)
        matched = Image.fromarray(matched)
        matched.save(f"{outdir}/{i}.png")


def save_images(images, outdir):
    for i in range(len(images)):
        image = np.array(Image.open(images[i]).convert("L"))
        image = Image.fromarray(image)
        image.save(f"{outdir}/{i}.png")


def save_predicted_masks(masks, outdir):
    for i in range(len(masks)):
        image = Image.fromarray(masks[i])
        image.save(f"{outdir}/{i}.png")


def clean_folder(folder):
    files = glob.glob(os.path.join(folder, "*.png"))
    for file in files:
        os.remove(file)


def remove_background(
    input_path,
    output_path,
    background_color=(255, 255, 255),
    tolerance=30,
    new_background_color=(0, 0, 0),
):
    # Open the image
    img = Image.open(input_path).convert("RGBA")

    # Convert image to numpy array
    np_img = np.array(img)

    # Create a mask based on the background color
    mask = np.all(np.abs(np_img[:, :, :3] - background_color) < tolerance, axis=2)

    # Create an alpha channel
    alpha = np.where(mask, 0, 255).astype(np.uint8)

    # Add the alpha channel to the image
    np_img[:, :, 3] = alpha

    # Convert back to PIL Image
    result = Image.fromarray(np_img)

    # Create a new image with the desired background color
    new_background = Image.new("RGBA", result.size, new_background_color + (255,))

    # Paste the result onto the new background using alpha compositing
    new_background.paste(result, (0, 0), result)
    new_background.save(output_path)

    return new_background


# out_df = pd.DataFrame(fetal_volumes.items())
# out_df.columns = ['case', 'fetal_volume']
# out_df['AF_volume'] = AF_volumes
# out_df['sac_volume'] = sac_volumes
# out_df['fafo'] = fafo_cases

# out_df.to_csv('all_cases_fafo_volumes_sac_5.csv', index = False)
