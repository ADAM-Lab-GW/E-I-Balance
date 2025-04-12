from generateHTML import *
from weightAnalysis import *
from getActivity import *

def writeAccSection(reportFile, dataSet, trialRange):
    # accuracy and loss section
    writeHTMLSectionHeader(reportFile, 'Accuracy and Loss', note='Accuracy and Loss curves can be combined onto single plots for easy comparison')
    accAndTags = []
    lossAndTags = []
    for i in trialRange:
        simTag = dataSet+'_'+str(i)+'_t'+str((i-100)%8)
        accTag = simTag+'_accList'
        lossTag = simTag+'_lossList'
        obj_path = '/lustre/groups/adamgrp/joey-ICONS-2023/surrogate-learning/data/'+dataSet+'/obj'
        accList = getWeightFromUniqueId(accTag, objFolder=obj_path)
        lossList = getWeightFromUniqueId(lossTag, objFolder=obj_path)

        accAndTags.append( (simTag,accList) )
        lossAndTags.append( (simTag,lossList) )

    writeAccTableDistributionHeader(reportFile)
    for accAndTag in accAndTags:
        (simTag,accList) = accAndTag
        writeAccDistributionRow(reportFile, simTag, np.array(accList))
    writeTableDistributionFooter(reportFile)

    accGraph, lossGraph = generateMultAccLossGraph(dataSet, accAndTags, lossAndTags)
    writeImageHTML(reportFile, accGraph, simTag=dataSet,descriptor='accuracy')

    writeLossTableDistributionHeader(reportFile)
    for lossAndTag in lossAndTags:
        (simTag,lossList) = lossAndTag
        writeLossDistributionRow(reportFile, simTag, np.array(lossList))
    writeTableDistributionFooter(reportFile)
    writeImageHTML(reportFile, lossGraph, simTag=dataSet,descriptor='loss')
    writeHTMLSectionFooter(reportFile)

def writeSectionEIRatio(reportFile, dataSet= 'sparse050_10class'):
    writeHTMLSectionHeader(reportFile, dataSet)
    descriptor = '_weights'
    writeHTMLSectionHeader(reportFile, 'Sample Weight Dist', note='This is a test of weight distributionss\n')
    writeTableDistributionHeader(reportFile)

    # sample weight distributions
    graphData = []
    for i in range(100, 200):
        try:
            # init training data
            simTag = dataSet+'_'+str(i)+'_t'+str((i-100)%8)+'_w1_init'
            print(simTag)
            objFolder = '/lustre/groups/adamgrp/joey-ICONS-2023/surrogate-learning/data/'+dataSet+'/obj'
            tmpW = getWeightFromUniqueId(simTag, objFolder=objFolder)

            writeDistributionRow(reportFile,simTag=simTag,notes='This is a test row', rawValuesNP=tmpW.flatten().detach().numpy())
            graphFile = generateHistogram(simTag, 'testing', tmpW.flatten().detach().numpy(), xlabel='Weight', ylabel='Frequency', numBins=20, xmin=None, xmax=None, ymin=None, ymax=None)

            graphData.append([graphFile, simTag, descriptor])

            # end of training data
            simTag = dataSet+'_'+str(i)+'_t'+str((i-100)%8)+'_w1_e29'
            print(simTag)
            objFolder = '/lustre/groups/adamgrp/joey-ICONS-2023/surrogate-learning/data/'+dataSet+'/obj'
            tmpW = getWeightFromUniqueId(simTag, objFolder=objFolder)

            writeDistributionRow(reportFile,simTag=simTag,notes='This is a test row', rawValuesNP=tmpW.flatten().detach().numpy())

            print('HANDLED '+str(i))
        except:
            print('CAUGHT FAILED DATA '+str(i))

    writeTableDistributionFooter(reportFile)

    writeHTMLSectionHeader(reportFile, 'Weight Graphs', note='all weight distributions, note that the x axis and y axis are not the same across simulations')
    for graphDataSet in graphData:
        writeImageHTML(reportFile, graphDataSet[0], graphDataSet[1], graphDataSet[2])
    writeHTMLSectionFooter(reportFile)

    # Sample activity
    for i in range(100,200,2):
        simTag = dataSet+'_'+str(i)+'_t'+str((i-100)%8)
        writeHTMLSectionHeader(reportFile, 'Sample Activity '+simTag, note='this is the aggregate of activity over time for networks')

        descriptor = 'network_activity'
        w1_tag = simTag+'_w1_init'
        w2_tag = simTag+'_w2_init'
        param_tag = simTag+'_surrParams'
        obj_path = '/lustre/groups/adamgrp/joey-ICONS-2023/surrogate-learning/data/'+dataSet+'/obj'
        try:
            params = getWeightFromUniqueId(param_tag, objFolder=obj_path)
            excClassActivity, inhClassActivity, isiByClass, distance = getActivityISIAndDistanceOverTime(w1_tag, w2_tag, params, obj_path)
        except:
            writeHTMLNote(reportFile, note='ERROR LOADING DATA AND CALCULATING ACTIVITY')
            writeHTMLSectionFooter(reportFile)
            continue
        
        try:
            # Spike time table of distributions by output class
            writeHTMLNote(reportFile, note='Excitatory spike time distributions')
            writeTableDistributionHeader(reportFile)
            for i in range(params.nb_outputs):
                writeDistributionRow(reportFile, simTag, excClassActivity[i], notes='Exc data for Class '+str(i))
            writeTableDistributionFooter(reportFile)
        except:
            writeHTMLNote(reportFile, note='EXCITATORY DATA NOT SHOWN BECAUSE OF LACK OF ACTIVITY')

        try:
            writeHTMLNote(reportFile, note='Inhibitory spike time distributions')
            writeTableDistributionHeader(reportFile)
            for i in range(params.nb_outputs):
                writeDistributionRow(reportFile, simTag, inhClassActivity[i], notes='Inh data for Class '+str(i))
            writeTableDistributionFooter(reportFile)
        except:
            writeHTMLNote(reportFile, note='INHIBITORY DATA NOT SHOWN BECAUSE OF LACK OF ACTIVITY')

        # ISI table of distributions by output class
        try:
            writeHTMLNote(reportFile, note='InterSpike Interval Distributions')
            writeTableDistributionHeader(reportFile)
            for i in range(params.nb_outputs):
                writeDistributionRow(reportFile, simTag, isiByClass[i], notes='ISI data for Class '+str(i))
            writeTableDistributionFooter(reportFile)
        except:
            writeHTMLNote(reportFile, note='INTERSPIKE INTERVAL DATA NOT SHOWN BECAUSE OF LACK OF ACTIVITY')

        # van rossum distance table
        writeHTMLNote(reportFile, note='Van Rossum Distance of Hidden Layer')
        writeTableDistributionHeader(reportFile)
        writeDistributionRow(reportFile, simTag, distance)
        writeTableDistributionFooter(reportFile)
        writeDistanceMatrixTable(reportFile, distance)

        # spike times and ISI graphs
        for i in range(params.nb_outputs):
            descriptor = 'network_activity_case_'+str(i)
            graphFile = generateSpikingActivity(simTag, descriptor, excClassActivity[i], inhClassActivity[i], ylabel='AVG Spikes Class '+str(i))
            writeImageHTML(reportFile, graphFile, simTag, descriptor)
            descriptor = 'network_ISI_case_'+str(i)
            graphFile = generateISIDistribution(simTag, descriptor, isiByClass[i], ylabel='Total Frequency Class '+str(i), yscale='symlog')
            writeImageHTML(reportFile, graphFile, simTag, descriptor)

        writeHTMLNote(reportFile, note='POST TRAINING DATA')
        w1_tag = simTag+'_w1_e29'
        w2_tag = simTag+'_w2_e29'
        param_tag = simTag+'_surrParams'
        obj_path = '/lustre/groups/adamgrp/joey-ICONS-2023/surrogate-learning/data/'+dataSet+'/obj'
        try:
            params = getWeightFromUniqueId(param_tag, objFolder=obj_path)
            excClassActivity, inhClassActivity, isiByClass, distance = getActivityISIAndDistanceOverTime(w1_tag, w2_tag, params, obj_path)
        except:
            writeHTMLNote(reportFile, note='ERROR LOADING DATA AND CALCULATING ACTIVITY')
            writeHTMLSectionFooter(reportFile)
            continue
        
        try:
            # Spike time table of distributions by output class
            writeHTMLNote(reportFile, note='Excitatory spike time distributions')
            writeTableDistributionHeader(reportFile)
            for i in range(params.nb_outputs):
                writeDistributionRow(reportFile, simTag, excClassActivity[i], notes='Exc data for Class '+str(i))
            writeTableDistributionFooter(reportFile)
        except:
            writeHTMLNote(reportFile, note='EXCITATORY DATA NOT SHOWN BECAUSE OF LACK OF ACTIVITY')

        try:
            writeHTMLNote(reportFile, note='Inhibitory spike time distributions')
            writeTableDistributionHeader(reportFile)
            for i in range(params.nb_outputs):
                writeDistributionRow(reportFile, simTag, inhClassActivity[i], notes='Inh data for Class '+str(i))
            writeTableDistributionFooter(reportFile)
        except:
            writeHTMLNote(reportFile, note='INHIBITORY DATA NOT SHOWN BECAUSE OF LACK OF ACTIVITY')

        # ISI table of distributions by output class
        try:
            writeHTMLNote(reportFile, note='InterSpike Interval Distributions')
            writeTableDistributionHeader(reportFile)
            for i in range(params.nb_outputs):
                writeDistributionRow(reportFile, simTag, isiByClass[i], notes='ISI data for Class '+str(i))
            writeTableDistributionFooter(reportFile)
        except:
            writeHTMLNote(reportFile, note='INTERSPIKE INTERVAL DATA NOT SHOWN BECAUSE OF LACK OF ACTIVITY')

        # van rossum distance table
        writeHTMLNote(reportFile, note='Van Rossum Distance of Hidden Layer')
        writeTableDistributionHeader(reportFile)
        writeDistributionRow(reportFile, simTag, distance)
        writeTableDistributionFooter(reportFile)
        writeDistanceMatrixTable(reportFile, distance)

        # spike times and ISI graphs
        for i in range(params.nb_outputs):
            descriptor = 'network_activity_case_'+str(i)
            graphFile = generateSpikingActivity(simTag, descriptor, excClassActivity[i], inhClassActivity[i], ylabel='AVG Spikes Class '+str(i))
            writeImageHTML(reportFile, graphFile, simTag, descriptor)
            descriptor = 'network_ISI_case_'+str(i)
            graphFile = generateISIDistribution(simTag, descriptor, isiByClass[i], ylabel='Total Frequency Class '+str(i), yscale='symlog')
            writeImageHTML(reportFile, graphFile, simTag, descriptor)
        writeHTMLSectionFooter(reportFile)

        writeHTMLSectionFooter(reportFile)

    # accuracy and loss section
    writeHTMLSectionHeader(reportFile, 'Accuracy and Loss', note='Accuracy and Loss curves can be combined onto single plots for easy comparison')
    accAndTags = []
    lossAndTags = []
    for i in range(100,200,2):
        simTag = dataSet+'_'+str(i)+'_t'+str((i-100)%8)
        accTag = simTag+'_accList'
        lossTag = simTag+'_lossList'
        obj_path = '/lustre/groups/adamgrp/joey-ICONS-2023/surrogate-learning/data/'+dataSet+'/obj'
        try:
            accList = getWeightFromUniqueId(accTag, objFolder=obj_path)
            lossList = getWeightFromUniqueId(lossTag, objFolder=obj_path)
        except:
            writeHTMLNote(reportFile, note='COULD NOT LOAD '+simTag)
            continue

        accAndTags.append( (simTag,accList) )
        lossAndTags.append( (simTag,lossList) )

    writeAccTableDistributionHeader(reportFile)
    for accAndTag in accAndTags:
        (simTag,accList) = accAndTag
        writeAccDistributionRow(reportFile, simTag, np.array(accList))
    writeTableDistributionFooter(reportFile)

    accGraph, lossGraph = generateMultAccLossGraph(dataSet, accAndTags, lossAndTags)
    writeImageHTML(reportFile, accGraph, simTag=dataSet,descriptor='accuracy')

    writeLossTableDistributionHeader(reportFile)
    for lossAndTag in lossAndTags:
        (simTag,lossList) = lossAndTag
        writeLossDistributionRow(reportFile, simTag, np.array(lossList))
    writeTableDistributionFooter(reportFile)
    writeImageHTML(reportFile, lossGraph, simTag=dataSet,descriptor='loss')
    writeHTMLSectionFooter(reportFile)

    writeHTMLSectionFooter(reportFile)

reportFile = './TEST-REPORT-095.html'
writeHTMLHeader(reportFile)

#writeSectionEIRatio(reportFile, dataSet= 'sparse050_10class')
#writeSectionEIRatio(reportFile, dataSet= 'sparse080_10class')
writeSectionEIRatio(reportFile, dataSet= 'sparse095_10class')
#writeSectionEIRatio(reportFile, dataSet= 'sparse100_10class')

writeHTMLFooter(reportFile)

# plotWeightTraj(simTag, dataSet, descriptor)
# simTag = 'sparse080_10class_112_t4'
# dataSet = 'sparse080_10class'
# plotWeightTraj(simTag, dataSet, descriptor)
# simTag = 'sparse095_10class_115_t7'
# dataSet = 'sparse095_10class'
# plotWeightTraj(simTag, dataSet, descriptor)
# simTag = 'sparse100_10class_111_t3'
# dataSet = 'sparse100_10class'
# plotWeightTraj(simTag, dataSet, descriptor)
