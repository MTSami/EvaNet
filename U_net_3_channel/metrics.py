import statistics as stats
import numpy as np
from sklearn.metrics import confusion_matrix



class Evaluator():
    """
    The Evaluator currently can do the following metrics:
        - Precision
        - Recall
        - Fscore
    """

    def __init__(self):

        # Declare Metrics
        self.DRY_ACC = 0
        self.FLOOD_ACC = 0
        
        self.DRY_PRECISION = 0
        self.FLOOD_PRECISION = 0
        
        self.DRY_RECALL = 0
        self.FLOOD_RECALL = 0
        
        self.DRY_FSCORE = 0
        self.FLOOD_FSCORE = 0
    
    def run_eval(self, pred_unpadded, gt_labels):
        
        cm = confusion_matrix(gt_labels.flatten(), pred_unpadded.flatten(), labels = [0, 1, -1])
        TP_0 = cm[0][0]
        FP_0 = cm[1][0]
        FN_0 = cm[0][1]
        TN_0 = cm[1][1]
        
        
        TP_1 = cm[1][1]
        FP_1 = cm[0][1]
        FN_1 = cm[1][0]
        TN_1 = cm[0][0]
        
        
        ####DRY
        self.DRY_ACC = ((TP_0+TN_0)/(TP_0+TN_0+FP_0+FN_0))*100
        print("Dry Accuracy: ", self.DRY_ACC)
        self.DRY_PRECISION = ((TP_0)/(TP_0+FP_0))*100
        print("Dry Precision: ", self.DRY_PRECISION)
        self.DRY_RECALL = ((TP_0)/(TP_0+FN_0))*100
        print("Dry Recall: ", self.DRY_RECALL)
        self.DRY_FSCORE = ((2*self.DRY_PRECISION*self.DRY_RECALL)/(self.DRY_PRECISION+self.DRY_RECALL))
        print("Dry F-score: ", self.DRY_FSCORE)
        
        print("\n")
        
        ####FLOOD
        self.FLOOD_ACC = ((TP_1+TN_1)/(TP_1+TN_1+FP_1+FN_1))*100
        print("Flood Accuracy: ", self.FLOOD_ACC)
        self.FLOOD_PRECISION = ((TP_1)/(TP_1+FP_1))*100
        print("Flood Precision: ", self.FLOOD_PRECISION)
        self.FLOOD_RECALL = ((TP_1)/(TP_1+FN_1))*100
        print("Flood Recall: ", self.FLOOD_RECALL)
        self.FLOOD_FSCORE = ((2*self.FLOOD_PRECISION*self.FLOOD_RECALL)/(self.FLOOD_PRECISION+self.FLOOD_RECALL))
        print("Flood F-score: ", self.FLOOD_FSCORE)

        
    
    
    @property
    def f_accuracy(self):        
        if self.FLOOD_ACC > 0:
            return self.FLOOD_ACC
        else:
            return 0.0

    @property
    def f_precision(self):        
        if self.FLOOD_PRECISION > 0:
            return self.FLOOD_PRECISION
        else:
            return 0.0

 
    @property
    def f_recall(self):
        if self.FLOOD_RECALL > 0:
            return self.FLOOD_RECALL
        else:
            return 0.0
        
        
    @property
    def f_fscore(self):
        if self.FLOOD_FSCORE > 0:
            return self.FLOOD_FSCORE
        else:
            return 0.0
    
    
    
    
    @property
    def d_accuracy(self):        
        if self.DRY_ACC > 0:
            return self.DRY_ACC
        else:
            return 0.0
    
    @property
    def d_precision(self):        
        if self.DRY_PRECISION > 0:
            return self.DRY_PRECISION
        else:
            return 0.0

 
    @property
    def d_recall(self):
        if self.DRY_RECALL > 0:
            return self.DRY_RECALL
        else:
            return 0.0
        
        
    @property
    def d_fscore(self):
        if self.DRY_FSCORE > 0:
            return self.DRY_FSCORE
        else:
            return 0.0
