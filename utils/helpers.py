import os


def register_images(
    wf_ald,
    wf_res,
    reg_ald,
    resample,
    affine_path,
    source_path,
    target_path,
    ald_output_path,
    resample_data=True,
    resample_adc_mask=False,
):
    reg_ald.inputs.ref_file = source_path
    reg_ald.inputs.flo_file = target_path
    reg_ald.inputs.res_file = ald_output_path
    reg_ald.inputs.rig_only_flag = True
    reg_ald.inputs.nac_flag = True
    # reg_ald.gpuid_val = 2  # Use CPU
    # reg_ald.inputs.omp_core_val = 8
    reg_ald.inputs.aff_file = affine_path
    reg_ald.inputs.platform_val = 1
    wf_ald.run()

    if resample_data:
        resample.inputs.ref_file = source_path
        resample.inputs.flo_file = target_path
        resample.inputs.inter_val = "LIN"
        resample.inputs.trans_file = affine_path
        # resample.inputs.omp_core_val = 8
        resample.inputs.out_file = ald_output_path
        wf_res.run()

    if resample_adc_mask:
        print("Resampling ADC mask")
        adc_mask_path = target_path.replace("adc", "adc_masks")
        adc_mask_reg_path = ald_output_path.replace("adc_reg", "adc_reg_masks")
        if os.path.exists(adc_mask_path) and "adc" in adc_mask_path:
            resample_mask_with_transform(wf_res, resample, affine_path, source_path, adc_mask_path, adc_mask_reg_path)


def resample_mask_with_transform(wf_res, resample, affine_path, ref_path, mask_path, out_mask_path):
    """
    Resample a mask using a given transform, with nearest neighbor interpolation.
    """
    resample.inputs.ref_file = ref_path
    resample.inputs.flo_file = mask_path
    resample.inputs.inter_val = "NN"  # Nearest neighbor for masks
    resample.inputs.trans_file = affine_path
    # resample.inputs.omp_core_val = 8
    resample.inputs.out_file = out_mask_path
    wf_res.run()
