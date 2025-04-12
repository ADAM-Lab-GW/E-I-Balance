from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
from analysisUtil import *
from surrogateParams import surrParams
import pandas as pd
from guppy import hpy
from multiprocessing.pool import Pool
import os

h = hpy()
print(h.heap())

# get list of valid simtags
reportFolder = './reportHTML_VRD_fix/'
simTags = []
allParams = {}
for t in range(100, 252):
    for exc in [50, 80, 95, 100]:
        simTag = 'sparse{:03d}'.format(exc)+'_10class_'+str(t)+'_t'+str((t-100)%8)

        try:
            allParams[simTag] = getWeightFromUniqueId(simTag+'_surrParams', objFolder='./sparseAll_10class_surrParams')
            allReports = True
            for e in [-1,29]:
                reportFile = reportFolder+simTag+'_e'+str(e)+'.html'
                if not os.path.exists(reportFile):
                    print('MISSING REPORT '+reportFile)
                    allReports = False
                    break
            if not allReports: continue

            simTags.append(simTag)
            
        except Exception as e:
            print('SKIP: '+simTag+'\tEXCEPTION: '+str(e))


print('LOADING SIMTAGS:')
print(simTags)
input('PRESS ENTER TO CONINTUE')

# method for multiprocessing aggregation of all data from reports
def rowProcessingThreadAll(simTag):
    print('HANDLING SIMTAG: '+simTag)
    tmpRow = {}
    reportFolder = './reportHTML_VRD_fix/'
    try:
        # parameter data
        params = getWeightFromUniqueId(simTag+'_surrParams', objFolder='./sparseAll_10class_surrParams')
        tmpRow.update(vars(params))

        # confusion matrix data
        for e in [-1,29]:
            TP, TN, FP, FN, sensitivity, specificity = generateConfusionAnalysis(simTag, reportFolder=reportFolder, epoch=e)
            for c in range(10):
                tmpRow['TP_e'+str(e)+'_c'+str(c)] = TP[c][e+1]
                tmpRow['TN_e'+str(e)+'_c'+str(c)] = TN[c][e+1]
                tmpRow['FP_e'+str(e)+'_c'+str(c)] = FP[c][e+1]
                tmpRow['FN_e'+str(e)+'_c'+str(c)] = FN[c][e+1]

        # raw activity (# spikes)
        for e in [-1,29]:
            exc, inh = getRawActivity(simTag,reportFolder=reportFolder, skipGraphs=True, epoch=e)
            for c in range(10):
                tmpRow['excActAvg_e'+str(e)+'_c'+str(c)] = exc[e+1,c]
                tmpRow['inhActAvg_e'+str(e)+'_c'+str(c)] = inh[e+1,c]

        # particular data is loaded from file, meaning it is handled on
        #  a per epoch level
        for e in [-1,29]:
            simTagFull = simTag+'_e'+str(e)
            
            # normalized activity
            nExc = int(params.nb_hidden*params.excHidRat)
            nInh = params.nb_hidden - nExc
            avgInh, avgExc, inhLst, excLst = getNormActivity(reportFolder+simTagFull+'.html', nInh=nInh, nExc=nExc)
            tmpRow['normExcActAvg_e'+str(e)] = avgExc
            tmpRow['normInhActAvg_e'+str(e)] = avgInh
            for c in range(10):
                tmpRow['normExcAct_e'+str(e)+'_c'+str(c)] = excLst[c]
                tmpRow['normInhAct_e'+str(e)+'_c'+str(c)] = inhLst[c]
  
            # get the acc and loss data
            if e == -1:
                accTmp, lossTmp = getAccStats(reportFolder+simTagFull+'.html')
                tmpRow['maxAcc'] = float(accTmp[0]['Max Acc'])
                tmpRow['minLoss'] = float(lossTmp[0]['Max Loss'])

        # assuming all data could be loaded, we can return
        #  the entire row to be aggreagted for the dataframe
        return tmpRow        

    except Exception as e:
        print('SKIP: '+simTag+'\tEXCEPTION: '+str(e))
        return None

#rows = []
#with Pool(10) as pool:
#    for result in pool.map(rowProcessingThreadAll, simTags):
#        if result!=None:
#            rows.append(result)

#df = pd.DataFrame(rows)
#print(df)
#df.to_csv('./combinedEIData_multiprocessed_VRD_fix.csv')
df = pd.read_csv('./combinedEIData_multiprocessed_final.csv')

# handling van rossum data
def rowProcessingThread(simTag):
    global allParams
    try:
        tmpRow = {}
        param = allParams[simTag]
        #tmpRow.update(vars(param))
        tmpRow['simTag'] = simTag
        nExc = int(param.nb_hidden*param.excHidRat)
        print('simTag: '+simTag+'\tnExc: '+str(nExc))

        for e in [-1,29]:
            data = vanRossumDistanceFromSimTag(simTag, reportFolder='./reportHTML_VRD_fix/', epoch=e)
            EEAvg, EIAvg, IIAvg, EESTD, EISTD, IISTD = getVanStats(data, simTag, nExc=nExc, nInh=param.nb_hidden-nExc)
            tmpRow['e-e_avg_e'+str(e)] = EEAvg[0]
            tmpRow['e-i_avg_e'+str(e)] = EIAvg[0]
            tmpRow['i-i_avg_e'+str(e)] = IIAvg[0]
            tmpRow['e-e_std_e'+str(e)] = EESTD[0]
            tmpRow['e-i_std_e'+str(e)] = EISTD[0]
            tmpRow['i-i_std_e'+str(e)] = IISTD[0]
        
        return tmpRow    
    except Exception as e:
        print('SKIP: '+simTag+'\tEXCEPTION: '+str(e))
        return None


rows = []
with Pool(6) as pool:
    for result in pool.map(rowProcessingThread, simTags):
        if result!=None:
            rows.append(result)

vanDF = pd.DataFrame(rows)
print(vanDF)
vanDF.to_csv('./vanRossumDistanceData_VRD_fix.csv')

allDF = pd.merge(df, vanDF, on='simTag')
print(allDF)
allDF.to_csv('./combinedData_VRD_fix.csv')
