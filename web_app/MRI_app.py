import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from app_utils import (
    process_files,
    save_predicted_masks,
    clean_folder,
    save_images,
    read_case_files,
    create_masked_niftis,
    remove_background,
    nifti_to_stl,
)
from intpl_utils import transfer_folder, download_folder, run_remote_intpl
from PIL import Image
import subprocess

# Path to your Blender executable
BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe"

# Path to your main script
SCRIPT_PATH = r"glass_shader_main.py"


def run_blender_script():
    if not os.path.exists(BLENDER_PATH):
        raise FileNotFoundError(f"Blender executable not found at {BLENDER_PATH}")
    if not os.path.exists(SCRIPT_PATH):
        raise FileNotFoundError(f"Script not found at {SCRIPT_PATH}")

    # run Blender with the script
    command = [BLENDER_PATH, "--background", "--python", SCRIPT_PATH]

    # Run the command
    subprocess.run(command, check=True)


# Streamlit app
def app():
    st.title("Amniotic Fluid MRI Image Analysis")
    st.markdown(f"## MRIcroGL integrated")

    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if st.button("Run Analysis"):
            with st.spinner("Processing file..."):
                for x in range(len(df)):
                    (
                        AF_vol,
                        fafo_log,
                        fetal_vol,
                        diagnosis,
                        masks,
                        images,
                    ) = process_files(df, x)

                    output = {
                        "AF vol": f"{AF_vol: 0.2f} ml",
                        "FAFO_3d": f"{fafo_log: 0.2f}",
                        "Fetal vol": f"{fetal_vol:0.2f} ml",
                        "Diagnosis": diagnosis,
                    }

                    # Convert the dictionary to a pandas DataFrame
                    df2 = pd.DataFrame(
                        list(output.items()), columns=["Parameter", "Value"]
                    )

                    # create directory for output masks
                    masks_dir = "C:/Users/User/Downloads/app_masks"
                    if not os.path.exists(masks_dir):
                        os.makedirs(masks_dir)

                    # create directory for output masks
                    images_dir = "C:/Users/User/Downloads/app_case"
                    if not os.path.exists(images_dir):
                        os.makedirs(images_dir)

                    save_images(images, images_dir)
                    save_predicted_masks(masks, masks_dir)

                    image_paths = read_case_files(images_dir)
                    mask_paths = read_case_files(masks_dir)

                    output_path = "C:/Users/User/Downloads"
                    # png_output_dir = "C:/Users/User/Downloads/all_notebooks/png_output"
                    create_masked_niftis(image_paths, mask_paths, output_path)
                    # Convert a single file
                    nifti_to_stl(
                        input_path=r"C:\Users\User\Downloads\fluid.nii",
                        output_path=r"C:\Users\User\Downloads\fluid.stl",
                        threshold=0.5,
                        smoothing=1,
                    )

                    # The command you want to execute

                    subprocess.call(
                        [
                            "C:/Users/User/Downloads/MRIcroGL_windows/MRIcroGL/MRIcroGL.exe",
                            "C:/Users/User/Downloads/fetus.py",
                        ]
                    )
                    # subprocess.call(['C:/Users/User/Downloads/MRIcroGL_windows/MRIcroGL/MRIcroGL.exe','C:/Users/User/Downloads/fluid.py'])
                    run_blender_script()

                    AF1 = Image.open("C:/Users/User/Downloads/fluid_0.png")

                    input_image1 = "C:/Users/User/Downloads/fetus_0.png"
                    output_image1 = "C:/Users/User/Downloads/fetus_0_edited.png"
                    input_image2 = "C:/Users/User/Downloads/fetus_1.png"
                    output_image2 = "C:/Users/User/Downloads/fetus_1_edited.png"
                    input_image3 = "C:/Users/User/Downloads/fetus_2.png"
                    output_image3 = "C:/Users/User/Downloads/fetus_2_edited.png"

                    original_background_color = (41, 32, 32)  #
                    new_background_color = (0, 0, 0)  # Black background
                    tolerance = 30
                    fetus1 = remove_background(
                        input_image1,
                        output_image1,
                        original_background_color,
                        tolerance,
                        new_background_color,
                    )
                    fetus2 = remove_background(
                        input_image2,
                        output_image2,
                        original_background_color,
                        tolerance,
                        new_background_color,
                    )
                    fetus3 = remove_background(
                        input_image3,
                        output_image3,
                        original_background_color,
                        tolerance,
                        new_background_color,
                    )

                    # display case name
                    case_name = df.loc[x, "case"]

                    st.markdown(f"## Case {case_name}")

                    # Create two columns
                    # col1, col2, col3 = st.columns([1,2,2])

                    # Display the DataFrame as a table in the first column
                    st.table(df2)

                    # Create two columns inside the container
                    col1, col2 = st.columns(2)

                    # Display the images in the columns
                    with col1:
                        st.image(AF1, width=400, caption="AF 3D Volume")

                    with col2:
                        # Display the image in the second column

                        st.image(fetus1, width=400, caption="Fetus 3D Volume")

                    clean_folder(images_dir)
                    clean_folder(masks_dir)

    else:
        st.warning("Please upload a CSV file.")


if __name__ == "__main__":
    app()
