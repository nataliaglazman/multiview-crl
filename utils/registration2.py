# ...existing code...
import logging
import os
import sys

sys.path.insert(1, os.path.abspath("."))

import glob

from nipype import Node, Workflow
from nipype.interfaces import niftyreg
from tqdm.auto import tqdm
from utilities import init_logger

from helpers import register_images


def main():
    LOG_FILE = "logs/lesion_registration.log"
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    LOGGER = init_logger(LOG_FILE)

    # Set the log level to ERROR
    logging.getLogger("nipype.workflow").setLevel(logging.ERROR)

    base_folder = "/data/natalia/ADNI_nifti_reorient"
    # base_folder = "notebooks/xnat_processed"
    output_path = "ADNI_registered"
    os.makedirs(f"{output_path}", exist_ok=True)
    # use an absolute output path and ensure it exists and is writable
    output_path = os.path.abspath("ADNI_registered")
    os.makedirs(output_path, exist_ok=True)
    if not os.access(output_path, os.W_OK):
        LOGGER.error("Output path not writable: %s", output_path)
        # Optionally exit here if you want to fail fast
        # sys.exit(1)

    # create a RegAladdin node
    reg_ald = Node(niftyreg.RegAladin(), name="reg_aladdin")
    resample = Node(niftyreg.RegResample(), name="resample")

    reg_ald.interface.terminal_output = "none"
    resample.interface.terminal_output = "none"
    # stream command line so you can see the exact external invocation in logs
    reg_ald.interface.terminal_output = "stream"
    resample.interface.terminal_output = "stream"

    # Create a workflow
    wf_ald = Workflow(name="registration")
    wf_ald.base_dir = os.path.abspath(base_folder)
    # Add the node to the workflow
    wf_ald.add_nodes([reg_ald])

    wf_res = Workflow(name="resample")
    wf_res.base_dir = os.path.abspath(base_folder)
    wf_res.add_nodes([resample])

    LOGGER.info("Workflow created")
    LOGGER.info("Starting registration")

    for label in tqdm(os.listdir(f"{base_folder}"), total=len(os.listdir(f"{base_folder}"))):
        label = str(label)
        print(f"Registering {label} to MNI")

        mni_path = "/home/ng24/mri_preproc/mni/icbm_avg_152_t1_tal_lin.nii"
        mni_path = os.path.abspath(mni_path)
        t1_dir = os.path.join(base_folder, label, "t1")
        t1_dir = os.path.abspath(t1_dir)
        # look for both .nii and .nii.gz
        t1_candidates = sorted(glob.glob(os.path.join(t1_dir, "*.nii")) + glob.glob(os.path.join(t1_dir, "*.nii.gz")))
        t1_path = t1_candidates[0] if t1_candidates else print(f"No T1 found in {t1_dir}")
        # avoid using print() as a fallback (print returns None)
        t1_path = t1_candidates[0] if t1_candidates else None

        if t1_path:
            print(f"t1 path: {t1_path}")
        else:
            print(f"No T1 found in {t1_dir}")

        t2_dir = os.path.join(base_folder, label, "t2")
        t2_dir = os.path.abspath(t2_dir)
        t2_candidates = sorted(
            glob.glob(os.path.join(t2_dir, "*FLAIR*.nii")) + glob.glob(os.path.join(t2_dir, "*FLAIR*.nii.gz"))
        )
        t2_path = t2_candidates[0] if t2_candidates else print(f"No T2 found in {t2_dir}")
        t2_path = t2_candidates[0] if t2_candidates else None
        if t2_path:
            print(f"t2 path: {t2_path}")
        else:
            print(f"No T2 found in {t2_dir}")

        affine_path = os.path.join(output_path, "registration", "reg_aladdin", f"{label}_aff.txt")
        affine_path = os.path.abspath(affine_path)

        try:
            if t2_path and os.path.exists(t2_path):
                print("Registering T2")
                t2_file_name = os.path.basename(t2_path)
                t2_directory = os.path.join(output_path, f"{label}", "t2")
                t2_directory = os.path.abspath(t2_directory)
                os.makedirs(t2_directory, exist_ok=True)
                os.makedirs(t2_directory, exist_ok=True)
                ald_output_path = os.path.join(output_path, f"{label}", "t2", t2_file_name)
                ald_output_path = os.path.abspath(ald_output_path)
                # ensure parent dirs for affine  output exist & are writable
                os.makedirs(os.path.dirname(affine_path), exist_ok=True)
                os.makedirs(os.path.dirname(ald_output_path), exist_ok=True)

                print(f"t2 output filename: {t2_file_name}")

                t2_out_dir = os.path.join(output_path, label, "t2")

                print("WRITE PATH :", ald_output_path)
                print("SEARCH PATH:", t2_out_dir)
                print("DIR EXISTS :", os.path.exists(t2_out_dir))
                print("DIR CONTENTS:", os.listdir(t2_out_dir) if os.path.exists(t2_out_dir) else "MISSING")

                print(f"Looking for existing files in: {t2_out_dir}")

                possible_files = sorted(
                    glob.glob(os.path.join(t2_out_dir, "*FLAIR*.nii"))
                    + glob.glob(os.path.join(t2_out_dir, "*FLAIR*.nii.gz"))
                )

                print(f"Found existing outputs: {possible_files}")

                if possible_files:
                    LOGGER.info("Existing T2 output(s) found for %s, skipping: %s", label, possible_files)
                    continue

                if not os.access(os.path.dirname(ald_output_path), os.W_OK):
                    LOGGER.error("Output directory not writable: %s", os.path.dirname(ald_output_path))
                    continue
                # check if file already exists to avoid overwriting
                if os.path.exists(ald_output_path):
                    LOGGER.error(f"File already exists, skipping: {ald_output_path}")
                    continue
                register_images(wf_ald, wf_res, reg_ald, resample, affine_path, mni_path, t2_path, ald_output_path)

            # if t1_path and os.path.exists(t1_path):
            #     print('Registering T1')
            #     t1_file_name = os.path.basename(t1_path)
            #     t1_directory = os.path.join(output_path, f"{label}", "t1")
            #     t1_directory = os.path.abspath(t1_directory)
            #     os.makedirs(t1_directory, exist_ok=True)
            #     os.makedirs(t1_directory, exist_ok=True)
            #     ald_output_path = os.path.join(output_path, f"{label}", "t1", t1_file_name)
            #     ald_output_path = os.path.abspath(ald_output_path)
            #     os.makedirs(os.path.dirname(ald_output_path), exist_ok=True)
            #     if not os.access(os.path.dirname(ald_output_path), os.W_OK):
            #         LOGGER.error("Output directory not writable: %s", os.path.dirname(ald_output_path))
            #         continue
            #     register_images(
            #          wf_ald, wf_res, reg_ald, resample, affine_path, mni_path, t1_path, ald_output_path)
        except Exception as e:
            LOGGER.info(f"Error: {e}")
            LOGGER.info(f"Failed to register {label} t2")
            LOGGER.info(f"File paths: t2: {t2_path}")
            continue

        print(f"Finished registering {label}")

    LOGGER.info("Registration complete")


if __name__ == "__main__":
    main()
# ...existing code...
