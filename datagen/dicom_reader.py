import pylidc as pl
import statistics
import os
import numpy as np
import os
import sys

from skimage.transform import resize

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Scanotation():
    ''' Handles the LIDC-IDRI​ Dicom files. 
        args:
            number: Patient ID
    '''

    def __init__(self, number) -> None:
        number =  str(number).zfill(4)
        pid = 'LIDC-IDRI-{}'.format(number)
        self.scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
        self.nods = self.scan.cluster_annotations()
        # print("%s has %d nodules." % (self.scan, len(self.nods)))
        
        self.no_nodules = len(self.nods)

        self.vol = None

    def center(self, nodule_idx = 0):
        ''' Returns the nodule location.
            args:
                nodule_idx: Index of nodule, default: 0. 
                Can be >0 if there are more than one nodules for this recording.
        '''
        centroids = []
        for i in range(len(self.nods[nodule_idx])):
            centroids.append(self.nods[nodule_idx][i].centroid)
        return np.round(np.mean(centroids,axis=0))
    
    ## new
    def frames(self,  nodule_idx = 0):
        return self.nods[nodule_idx][0].contour_slice_indices

    def bbox(self, nodule_idx = 0):
        ''' Returns the nodule location.
            args:
                nodule_idx: Index of nodule, default: 0. 
                Can be >0 if there are more than one nodules for this recording.
        '''
        return self.nods[nodule_idx][0].bbox()

    def load_volume(self):
        ''' Loads the 3D volume of the scan. (No need to run this outside of the class).
        '''
        #blockPrint()
        self.vol = self.scan.to_volume()
        #enablePrint()

    def get_volume(self):
        ''' Returns the 3D volume of the scan.
        '''
        if self.vol is None:
            self.load_volume()
        return self.vol

    def nodule_slice(self, nodule_idx = 0):
        ''' Returns the 2D slice of the scan with a nodule in it.
            args:
                nodule_idx: Index of nodule, default: 0.
        '''
        slice_no = int(self.center(nodule_idx)[-1])
        if self.vol is None:
            self.load_volume()
        return self.vol[:,:,slice_no]   

    def nodule_crop(self, nodule_idx = 0):
        ''' Returns the 2D slice of the scan with a nodule in it, cropped around the nodule.
            args:
                nodule_idx: Index of nodule, default: 0.
        '''
        if self.vol is None:
            self.load_volume()
        return self.vol[self.bbox(nodule_idx)]

    def malignancy(self, nodule_idx = 0):
        ''' Returns the malignancy of the nodule.
            args:
                nodule_idx: Index of nodule, default: 0.
        '''
        nods = self.scan.cluster_annotations()
        mals = []
        for j in range(len(nods[nodule_idx])):
            #print('----> Malignancy', nods[nodule_idx][j].malignancy, nods[nodule_idx][j].Malignancy)
            mals.append(nods[nodule_idx][j].malignancy)
        diagnosis=(statistics.mean(mals)) #median_high
        if diagnosis>=4:
            malignancy_th = 1
        elif diagnosis<=2:
            malignancy_th = 0
        else: 
            malignancy_th = -1
        return malignancy_th,diagnosis


if __name__ == '__main__':
    scan=Scanotation(148)
    all_n = []
    for i in range((scan.no_nodules)):
        # print(scan.center(i))
        print(scan.center(i))
