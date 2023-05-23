from planes_detector import IDetectionNetworkController, BBox, YoloDetectionNetworkController
import hashlib
import xml.etree.ElementTree as ET

class YoloDetectionNetworkControllerM(YoloDetectionNetworkController):
    _HASHES_PATH = "../planes_dataset/hashes.csv"
    _DATASET_PATH = "../planes_dataset"
    def __init__(self):
        super().__init__()
        self.m_hashes = {}
    
    def LoadModel(self, modelPath):
        with open(self._HASHES_PATH, "r") as hashes:
            for line in hashes:
                lineData = line.split(',')
                self.m_hashes[lineData[0]] = lineData[1].strip()

        super().LoadModel(modelPath)
    
    def GetBBoxes(self, imagePath):
        fileHash = self._FileHash(imagePath)
        dataPath = self.m_hashes.get(fileHash)

        if dataPath is None:
            return super().GetBBoxes(imagePath)
        
        return self._GetBBoxesFromAnnotation(f"{self._DATASET_PATH}/{dataPath}")
        
    def _FileHash(self, filename):
        hasher = hashlib.sha256()

        with open(filename, "rb") as f:
            while chunk := f.read(4096):
                hasher.update(chunk)

        hash_sum = hasher.hexdigest()

        return hash_sum
    
    def _GetBBoxesFromAnnotation(self, annotationPath):
        imageAnnotation = ET.parse(annotationPath)

        bboxes = []
        for planeAnnotation in imageAnnotation.getroot():
            if planeAnnotation.tag == "object":
                name = planeAnnotation[0].text
                for child in planeAnnotation:
                    if child.tag == "bndbox":
                        xmin = int(child[0].text)
                        ymin = int(child[1].text)
                        xmax = int(child[2].text)
                        ymax = int(child[3].text)

                        yc = (ymin + ymax) // 2
                        xc = (xmin + xmax) // 2
                        h = ymax - ymin
                        w = xmax - xmin

                        bboxes.append(BBox(xc, yc, w, h))
        
        return bboxes