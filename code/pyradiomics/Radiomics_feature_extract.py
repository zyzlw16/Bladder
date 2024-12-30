# -*- coding: UTF-8 -*-
import argparse
import pandas as pd
import csv
import logging
import os
import shutil
import tempfile
import threading
import signal
import numpy as np
import SimpleITK as sitk
from datetime import datetime
from collections import OrderedDict
from multiprocessing import cpu_count, Pool
import radiomics
import glob
#import cv2

TEMP_DIR = tempfile.mkdtemp()

def window(img, img_path, output_base_dir='./ZJHP_HCC303_output_CT_P_images_60_170'):

    win_min = 37.5
    win_max = 425

    for i in range(img.shape[0]):
        img[i] = 255.0 * (img[i] - win_min) / (win_max - win_min)
        min_index = img[i] < 0
        img[i][min_index] = 0
        max_index = img[i] > 255
        img[i][max_index] = 255
        img[i] = img[i] - img[i].min()
        
        if img[i].max() != 0:
            c = float(255) / img[i].max()
        else:
            c = 1.0
        img[i] = img[i] * c


    return img.astype(np.uint8)

class info_filter(logging.Filter):
    def __init__(self, name):
        super(info_filter, self).__init__(name)
        self.level = logging.WARNING

    def filter(self, record):
        if record.levelno >= self.level:
            return True
        if record.name == self.name and record.levelno >= logging.INFO:
            return True
        return False


def get_compact_range(mask_arr):
    if np.sum(mask_arr) < 1:
        raise ValueError('Zero mask.')
    xyz = np.nonzero(mask_arr)
    xmin = np.min(xyz[0])
    xmax = np.max(xyz[0])
    ymin = np.min(xyz[1])
    ymax = np.max(xyz[1])
    zmin = np.min(xyz[2])
    zmax = np.max(xyz[2])
    result = ([xmin, xmax], [ymin, ymax], [zmin, zmax])
    return result

def extract_feature(case):
    feature_vector = OrderedDict(case)

    try:
        img_dir = case['image']
        mask_nii = case['mask']
        use_dicom = case['use_dicom']
        img_reader = case['img_reader']
        use_pyradiomics = case['use_pyradiomics']

        if use_pyradiomics:
            from radiomics.featureextractor import RadiomicsFeatureExtractor

        threading.current_thread().name = mask_nii

        case_id = mask_nii.replace('/', '_')

        filename = r'features_' + str(case_id) + '.csv'
        output_filename = os.path.join(TEMP_DIR, filename)

        t = datetime.now()

        # mask
        single_reader = sitk.ImageFileReader()
        single_reader.SetFileName(mask_nii)
        mask = single_reader.Execute()
        mask_arr = sitk.GetArrayFromImage(mask).astype(np.uint8)
        #mask_arr = np.where(mask_arr > 1, 0, mask_arr)
        mask_arr = np.where(mask_arr != 0, 1, mask_arr)
        # 如果mask有很少的voxel，则略过
        voxels = np.sum(mask_arr)
        if voxels <= 3:
            delta_t = datetime.now() - t
            logging.getLogger('radiomics_s.batch').error('Case: %s %s %s processed in %s PID %s (%s)',
                                                       case_id, case["image"], case["mask"], delta_t, os.getpid(),
                                                       "Mask only contains few segmented voxel! Ignored.")
            return feature_vector

        # 读取image
        if use_dicom:
            dicom_reader = sitk.ImageSeriesReader()
            dicom_reader.SetFileNames(dicom_reader.GetGDCMSeriesFileNames(img_dir))
            try:
                img = dicom_reader.Execute()
            except:
                fnames = glob.glob(img_dir + '/*.dcm')

                dicom_reader.SetFileNames(fnames)
        else:
            single_reader.SetFileName(img_dir)
            if "nrrd" in img_reader:
                single_reader.SetImageIO("NrrdImageIO")
            else:
                single_reader.SetImageIO("NiftiImageIO")
            img = single_reader.Execute()

        img_arr = sitk.GetArrayFromImage(img)
        # img_arr = window(img_arr)
        img_arr = window(img_arr, img_dir)
        img_arr = np.transpose(img_arr, (2, 1, 0))
        mask_arr = np.transpose(mask_arr, (2, 1, 0))


        radiomics.setVerbosity(logging.INFO)
        # Get the PyRadiomics logger (default log-level = INFO)
        logger = radiomics.logger
        # set level to DEBUG to include debug log messages in log file
        logger.setLevel(logging.DEBUG)
        # Set up the handler to write out all log entries to a file
        handler = logging.FileHandler(filename='pyrad_log.txt', mode='w')
        formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

     
        settings = {}
        # settings['binWidth'] = 5
        settings['resampledPixelSpacing'] = None  # [3,3,3]

        extractor = RadiomicsFeatureExtractor(**settings)

        if use_pyradiomics:

            extractor.enableAllFeatures()
            extractor.enableAllImageTypes()
            extractor.enableImageTypeByName('LoG', customArgs={'sigma': [1.0, 3.0, 5.0]})

        valid_range_z, valid_range_y, valid_range_x = get_compact_range(mask_arr)

        mask_arr = mask_arr[
                   valid_range_z[0]: valid_range_z[1] + 1,
                   valid_range_y[0]: valid_range_y[1] + 1,
                   valid_range_x[0]: valid_range_x[1] + 1]

        img_arr = img_arr[
                  valid_range_z[0]: valid_range_z[1] + 1,
                  valid_range_y[0]: valid_range_y[1] + 1,
                  valid_range_x[0]: valid_range_x[1] + 1]

        mask_itk = sitk.GetImageFromArray(mask_arr)


        img_itk = sitk.GetImageFromArray(img_arr)
        img_itk.SetSpacing(img.GetSpacing())
        img_itk.SetOrigin(img.GetOrigin())

        mask.SetSpacing(img.GetSpacing())
        mask.SetOrigin(img.GetOrigin())

        mask_itk.SetSpacing(img.GetSpacing())
        mask_itk.SetOrigin(img.GetOrigin())

        signature = extractor.execute(img_itk, mask_itk)

        signature = OrderedDict(('_'.join(k.split('_')[1:] + k.split('_')[:1]), v) for k, v in signature.items())

        feature_vector.update(signature)
        with open(output_filename, 'w') as outputFile:
            writer = csv.DictWriter(outputFile, fieldnames=list(feature_vector.keys()), lineterminator='\n')
            writer.writeheader()
            writer.writerow(feature_vector)

        # Display message
        delta_t = datetime.now() - t

        logging.getLogger('radiomics.batch').error('Case: %s %s %s processed in %s PID %s ',
                                                   case_id, case["image"], case["mask"],
                                                   delta_t, os.getpid())

    except KeyboardInterrupt:
        print('parent interrupted')

    except Exception:
        logging.getLogger('radiomics_s.batch').error('Feature extraction failed!', exc_info=True)

    return feature_vector

def choose_feature(feature_file, use_pyradiomics=True):
    feature_classes = ['glcm',
                       'gldm',
                       'glrlm',
                       'glszm',
                       'ngtdm',
                       'shape',
                       'firstorder']

    if not use_pyradiomics:
        feature_classes = [
            "glcm",
            "glrlm",
            "shape",
            "firstorder"
        ]

    df = pd.read_csv(feature_file)
    columns = df.columns
    valid_columns = ['image', 'mask'] + [x for x in columns if len([y for y in feature_classes if y in x[:len(y)]]) > 0]
    return df[valid_columns]

def main(data_csv, output_path, lib, cpus, img_reader):
    np.seterr(invalid='raise')

    ROOT = os.path.dirname(os.path.realpath(__file__))
    use_dicom = "dicom" in img_reader
    logging.getLogger('radiomics.batch').debug('Logging init')
    use_pyradiomics = lib in "pyradiomics"
    threading.current_thread().name = 'Main'
    REMOVE_TEMP_DIR = True
    NUM_OF_WORKERS = int(cpus)
    if NUM_OF_WORKERS < 1:
        NUM_OF_WORKERS = 1
    NUM_OF_WORKERS = min(cpu_count() - 1, NUM_OF_WORKERS)

    def term(sig_num, addtion):
        # os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
        print('Extraction abort, being killed by SIGTERM')
        pool.terminate()
        pool.join()
        return

    pool = Pool(NUM_OF_WORKERS)

    #print("Main PID {0}".format(os.getpid()))

    signal.signal(signal.SIGTERM, term)

    try:
        data_df = pd.read_csv(data_csv)
        try:
            images_list = data_df['image'].tolist()
        except:
            images_list = data_df['dataset'].tolist()
        masks_list = data_df['image'].tolist()

        cases = [{
            'image': images_list[i],
            'mask': masks_list[i],
            'use_dicom': use_dicom,
            'img_reader': img_reader,
            'use_pyradiomics': use_pyradiomics
        } for i in range(len(images_list))]

        log = os.path.join(TEMP_DIR, 'log.txt')
        sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
        radio_logger = radiomics.logger
        log_handler = logging.FileHandler(filename=log, mode='a')
        log_handler.setLevel(logging.INFO)
        log_handler.setFormatter(logging.Formatter('%(levelname)-.1s: (%(threadName)s) %(name)s: %(message)s'))
        radio_logger.addHandler(log_handler)
        output_handler = radio_logger.handlers[0]  # Handler printing to the output
        output_handler.setFormatter(logging.Formatter('[%(asctime)-.19s] (%(threadName)s) %(name)s: %(message)s'))
        output_handler.setLevel(logging.INFO)  # Ensures that INFO messages are being passed to the filter
        output_handler.addFilter(info_filter('radiomics.batch'))
        logger = logging.getLogger('radiomics.batch')
        logger.info('Loaded %d jobs', len(cases))
        logger.info('Starting parralel pool with %d workers out of %d CPUs', NUM_OF_WORKERS, cpu_count())

        results = pool.map_async(extract_feature, cases).get(888888)
        c_f = output_path.split('/')[-1]
        if not os.path.exists(output_path.replace(c_f, '')):
            os.makedirs(output_path.replace(c_f, ''))
        try:
            # Store all results into 1 file
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=None)

            if REMOVE_TEMP_DIR:
                logger.info('success')
                logger.info('Removing temporary directory %s (contains individual case results files)', TEMP_DIR)
                shutil.rmtree(TEMP_DIR)

        except Exception:
            logger.error('Error storing results into single file!', exc_info=True)

        chosen_feature = choose_feature(output_path, use_pyradiomics=use_pyradiomics)
        chosen_feature.to_csv(output_path, index=False, encoding='utf-8')
    except (KeyboardInterrupt, SystemExit):
        print("...... Exit ......")
        pool.terminate()
        pool.join()
        return
    else:
        print("......end......")
        pool.close()

    print("System exit")
    pool.join()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv', help='image and mask columns', default='./pyradiomics/ROI/SDPH_output_file_path.csv')
    parser.add_argument('--output', help='feature output csv name', default='/data/Bladder/pyradiomics/ROI/SDPH_ROI_radiomics_features.csv')
    parser.add_argument('--lib', help='Pyradiomics', default='py')
    parser.add_argument('--cpus', help='cpu cores', type=int, default=16)
    parser.add_argument('--img_reader', help='dicom, nii, nrrd', default="nii")
    args = parser.parse_args()
    main(args.data_csv, args.output, args.lib.lower(), args.cpus, args.img_reader)


