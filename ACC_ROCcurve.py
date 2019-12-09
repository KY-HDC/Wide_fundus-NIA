from sklearn.metrics import confusion_matrix
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc
import sklearn

################################################

print(sklearn.__version__)
print(pd.__version__)

################################################

CHART_SIZE = (9,9)
DPI        = 600
FONT_SIZE  = 12
LW0        = 0.5
LW1        = 1
LW2        = 2
LW4        = 4

################################################

def modify(dataframe):
    del dataframe['YT']
    del dataframe['OX']
    eye = dataframe['JPG'][0].split('_')[1]
    if eye != 'NOR':
        dataframe['YT'] = dataframe['JPG'].map(lambda x:1 if eye in x else 0)
    else:
        dataframe['YT'] = dataframe['JPG'].map(lambda x:0 if eye in x else 1)
    dataframe['bool'] = dataframe.YT == dataframe.YP
    dataframe['OX'] = dataframe['bool'].map(int)
    del dataframe['bool']
    return dataframe

################################################
    
def ROC_CURVE(filelist):
    for i in filelist:
        print(i)
        report = pd.read_csv(i, sep='\t')
        report.columns = ['EXP', 'FOLD', 'JPG','YT','YP','OX', 'LR', 'FILTER', 'SCALE', 'ROTATION', 'COLOR', 'P0', 'P1']
        modify(report) 
        
        disease_prob = report['P1']
        X_true = report['YT']
        X_pred = report['YP']
        confusion = confusion_matrix(X_true, X_pred)

        ACC = round((confusion[0][0] + confusion[1][1]) / (confusion[0][0] + confusion[0][1] + confusion[1][0] + confusion[1][1]),2) 
        Sensitivity = round(confusion[0][0] / (confusion[0][0] + confusion[1][0]),2)
        Specificity = round(confusion[1][1] / (confusion[1][1] + confusion[0][1]),2)
        print('ACC : {}'.format(ACC)) 
        print('Sensitivity : {}'.format(Sensitivity))
        print('Specificity : {}'.format(Specificity))
        
        y_labels = X_true 
        y_prob = disease_prob 
        fpr, tpr, thresholds = roc_curve(y_labels, y_prob) 
        
        plt.rcParams["figure.figsize"] = (5,4)
        plt.figure()
        plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
        plt.ylabel('True Positive Rate (recall)', fontsize=14)
        plt.title('Receiver operating characteristic (ROC) curve')
        plt.legend(loc="lower right")
        plt.show()
        
################################################
        
#ROC_CURVE(filelist) #실행 / filelist는 건양대 환경에 맞게 작성하셔야합니다.