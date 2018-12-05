import numpy as np
import datetime
import time
from collections import defaultdict
import copy
from pycocotools.cocoeval import COCOeval


class DAVISEval(COCOeval):
    def __init__(
            self, cocoGt=None, cocoDt=None, iouType='bbox'
    ):
        super(DAVISEval, self).__init__(cocoGt, cocoDt, iouType)
        self.params = ParamsDAVIS(iouType=iouType)
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            stats = np.zeros((16,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='extra_small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[9] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[12] = _summarize(0, areaRng='extra_small', maxDets=self.params.maxDets[2])
            stats[13] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[14] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[15] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()


class ParamsDAVIS:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2],
                        [0 ** 2, 32 ** 2],
                        [32 ** 2, 224 ** 2],
                        [224 ** 2, 387 ** 2],
                        [387 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'extra_small', 'small', 'medium', 'large']
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2],
                        [0 ** 2, 32 ** 2],
                        [32 ** 2, 224 ** 2],
                        [224 ** 2, 387 ** 2],
                        [387 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'extra_small', 'small', 'medium', 'large']
        self.useCats = 1

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None
