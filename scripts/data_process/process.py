import os
import pandas as pd
from multiprocessing import Pool, Manager, cpu_count
from functools import partial

import scipy.ndimage as ndi
import pydicom

import SimpleITK as sitk
from scipy.ndimage import zoom

import numpy as np
from collections import defaultdict
from tqdm import tqdm
from PIL import Image, ImageFile
import re


ImageFile.LOAD_TRUNCATED_IMAGES = True  # 
Image.MAX_IMAGE_PIXELS = None  # 

pydicom.config.use_gdcm = True

def load_dicom_series(dicom_dir):
    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_IDs:
        raise FileNotFoundError(f"No DICOM series found in {dicom_dir}")

    series_file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_IDs[0])
    reader.SetFileNames(series_file_names)
    image = reader.Execute()
    return sitk.GetArrayFromImage(image)  # shape: (depth, height, width)

def resize_volume(volume, target_shape=(32, 224, 224)):
    # Input shape: (D, H, W), output shape: (D', H', W')
    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
    resized = zoom(volume, zoom_factors)  # linear interpolation
    return resized

def get_orientation(file_path):
    ds = pydicom.dcmread(file_path, stop_before_pixels=True)
    try:
        iop = ds.ImageOrientationPatient
        if len(iop) != 6:
            return "Unknown"

        row_cosines = np.array(iop[:3])
        col_cosines = np.array(iop[3:])
        normal = np.cross(row_cosines, col_cosines)

        axis = np.argmax(np.abs(normal))
        if axis == 0:
            return "Sagittal"
        elif axis == 1:
            return "Coronal"
        elif axis == 2:
            return "Axial"
        else:
            return "Unknown"
    except:
        return "Unknown"

def get_mri_modality(file_path: str) -> str:
    """
    """

    def _get_text(ds, tag, default=""):
        val = ds.get(tag, default)
        if val is None:
            return ""
        # ImageType
        if isinstance(val, (list, tuple)):
            return " ".join(str(x) for x in val).upper()
        return str(val).upper()

    def _get_b_value(ds, series_desc="", protocol_name=""):
        if (0x0018, 0x9087) in ds:
            return str(ds[(0x0018, 0x9087)].value)

        text = f"{series_desc} {protocol_name}"
        match = re.search(r"[bB]\s*=?\s*(\d+)", text)
        if match:
            return match.group(1)

        return None

    ds = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)

    series_desc     = _get_text(ds, (0x0008, 0x103E))   # SeriesDescription
    protocol_name   = _get_text(ds, (0x0018, 0x1030))   # ProtocolName
    image_type      = _get_text(ds, (0x0008, 0x0008))   # ImageType (MultiValue)
    sequence_name   = _get_text(ds, (0x0018, 0x0024))   # SequenceName
    scanning_seq    = _get_text(ds, (0x0018, 0x0020))   # ScanningSequence
    # seq_variant     = _get_text(ds, (0x0018, 0x0021))   # SequenceVariant
    # scan_opts       = _get_text(ds, (0x0018, 0x0022))   # ScanOptions
    # manufacturer    = _get_text(ds, (0x0008, 0x0070))   # Manufacturer
    contrast_agent  = _get_text(ds, (0x0018, 0x0010))   # Contrast/Bolus Agent
    contrast_flags  = " ".join([
        _get_text(ds, (0x0018, 0x1040)),  # Contrast/Bolus Volume
        _get_text(ds, (0x0018, 0x1041)),  # Contrast Flow Rate
        _get_text(ds, (0x0018, 0x1042)),  # Contrast Flow Duration
        _get_text(ds, (0x0018, 0x1044)),  # Contrast/Bolus Total Dose
    ])

    text = " ".join([series_desc, protocol_name, image_type, ])
    text = re.sub(r"\s+", " ", text).strip()

    def has(*patterns):
        return any(re.search(p, text) for p in patterns)

    flair_pat = (
        r"FLAIR", r"DARK[- _]?FLUID", r"FLUID[- _]?ATTENUATED",
        r"IR[- _]?FLAIR", r"T2[- _]?FLAIR", r"T2FLAIR",
        r"SPACE[- _]?FLAIR", r"FLAIR[- _]?SPACE"
    )

    # DWI / ADC / DTI
    adc_pat = (r"ADC", r"APPARENT[- _]?DIFFUSION[- _]?COEFFICIENT")
    dwi_pat = (
        r"DWI", r"DW", r"DIFF",
        r"TRACE(W)?", r"b=?0", r"b=?50", r"b=?400", r"b=?800", r"b=?1000",
        r"EP2D[- _]?DIFF", r"DWI[- _]?TRACE"
    )

    # SWI / QSM / T2*
    swi_pat = (r"SWI", r"SUSCEPTIBILITY", r"SWAN", r"VEN[- _]?BOLD")

    mprage_pat = (
        r"MPRAGE", r"MP[- _]?RAGE",
        r"FSPGR", r"IR[- _]?SPGR", r"BRAVO",
        r"T1[- _]?3D", r"3D[- _]?T1", r"TFE", r"T1W[- _]?3D"
    )

    t1_pat = (r"T1", r"T1W", r"T1WI", r"SPGR", r"GR[- _]?IR", r"T1_")
    t2_pat = (r"T2", r"T2W", r"T2WI", r"TSE", r"FSE", r"SPACE|CUBE|VISTA", r"T2_")

    is_post = False #has(*post_pat) and not has(*pre_neg_pat)
    if "+C" in text:
        is_post = True

    print(text)

    label = ''
    if has(*flair_pat):
        label += "FLAIR_"
        
    if has(*adc_pat) or "ADC" in image_type:
        label += "ADC_"

    if has(*dwi_pat):
        if 'DTI' not in text:
            label += "DWI_"

    if "ADC" in label or 'DWI' in "ADC":
        b = _get_b_value(ds)
        if b is not None:
            label += b + "_"

    if has(*swi_pat):
        label += "SWI_"

    if has(*mprage_pat):
        label += "T1-MPRAGE+C_" if is_post else "T1-MPRAGE_"

    if has(*t1_pat):
        label += "T1_"
        
    if has(*t2_pat):
        label += "T2_"

    if is_post:
        label += "+C_"
    if 'T1C' in file_path:
        label += "T1+C_"
    print(label)
    return label if label else "Unknown_"

def process_all_dirs(dicom_path, output_root, patient_id, count, seq):
    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(dicom_path)

    if not series_IDs:
        print(f"[!] No DICOM series found in {dicom_path}")
        return

    for idx, series_uid in enumerate(series_IDs):
        try:
            series_files = reader.GetGDCMSeriesFileNames(dicom_path, series_uid)
            reader.SetFileNames(series_files)

            modality = seq #get_mri_modality(series_files[0])
            ori = get_orientation(series_files[0])
            print(ori, modality)
            # if ori != 'Axial': continue
            if 'Unknown_' in modality: continue
            
            image = reader.Execute()
            array = sitk.GetArrayFromImage(image)  # shape: (slices, H, W)

            save_path = os.path.join(output_root, patient_id)
            os.makedirs(save_path, exist_ok=True)
            save_name = os.path.join(save_path, f"{patient_id}_{modality}{ori}_{idx}_{count}.npz")
            if array.shape[0] < 10:
                continue
            if ori == "Sagittal":
                array = np.transpose(array, (2, 1, 0))  # Z <- X
                ori = "Axial"
            elif ori == "Coronal":
                array = np.transpose(array, (1, 0, 2))  # Z <- Y
                ori = "Axial"
            elif ori == "Axial":
                pass
            else:
                continue  

            if os.path.exists(save_name):
                continue
            resized_volume = resize_volume(array, target_shape=(32, 224, 224))
            resized_volume = (resized_volume - resized_volume.min()) / (resized_volume.max() - resized_volume.min()+1e-8) * 255
            resized_volume = resized_volume.astype(np.uint8)
            np.savez_compressed(save_name, resized_volume)
            print(f"[✓] Saved: {save_name} | shape={array.shape}, modality={modality}")
        except Exception as e:
            print(f"[!] Failed to process {series_uid} in {dicom_path}: {e}")


input_root = "/data/to/Meningioma-SEG-CLASS"    # can be changed to other dataset
output_root = "/data/to/Meningioma-SEG-CLASS_output/"  
os.listdir(input_root)
count = 0
task_inputs_params = []

excel = pd.read_excel('/path_to/Meningioma_metadata.xlsx') # to help find correct modality
cls_name = '' 
input_cls = ''
for ii in range(len(excel)):
    case_id = str(excel.loc[ii, 'CaseID'])
    dir_name = os.path.join(
        # input_root, 
        # str(case_id),
        # str(excel.loc[ii, 'StudyFolder']),
        str(excel.loc[ii, 'SeriesPath'])
        )
    xulie = str(excel.loc[ii, 'Seq'])
    if xulie not in ['FLAIR', 'T1+C', 'T1', 'T2']:
        continue
    if not os.path.isdir(dir_name): continue
    print(f"Processing directory: {dir_name}")

    if len(os.listdir(dir_name)) < 10:
        print(f"Skipping {dir_name} due to insufficient files.")
        continue
    
    task_inputs_params.append((dir_name, output_root, f"{case_id}", count, xulie))
    count += 1
print(count, len(task_inputs_params))
print(task_inputs_params[0])
import pdb
pdb.set_trace()

with Pool(processes=cpu_count()) as pool:
   pool.starmap(process_all_dirs, task_inputs_params[:])

 
os.listdir(input_root)
count = 0
task_inputs_params = []

cls_name = ''
input_cls = ''
for sub_dir in os.listdir(input_root):
    dir_name = os.path.join(input_root, sub_dir)
    if not os.path.isdir(dir_name): continue

    print(f"Processing directory: {dir_name}")
    candidata = {}
    for root, dirs, files in os.walk(dir_name):
        for f in files:
            if f.endswith('.dcm'):
                candidata[root] = 0
    for d in candidata:
        if len(os.listdir(d)) < 10:
            print(f"Skipping {d} due to insufficient files.")
            continue
        if os.path.exists(os.path.join(output_root, f"{sub_dir}")):
            continue
        task_inputs_params.append((d, output_root, f"{sub_dir}", count))
        count += 1
print(count, len(task_inputs_params))
print(task_inputs_params[0])

with Pool(processes=cpu_count()) as pool:
   pool.starmap(process_all_dirs, task_inputs_params[:])
