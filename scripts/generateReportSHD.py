from generateHTML import *
from weightAnalysis import *
from getActivity import *
import argparse
def writeActivitySection(reportFile, simTag, objFolder, calcDistance=False, epoch=-1):
    # this section generates the metrics that cover the activity of the network
    #  each metric is often broken down by excitatory (exc) and inhibitory (inh) activity
    #  and is further broken down by class
    writeHTMLSectionHeader(reportFile, 'Sample Activity '+simTag+' epoch:'+str(epoch), note='this is the aggregate of activity over time for networks')
    
    descriptor = 'network_activity'
    if epoch==-1:
        w1_tag = simTag+'_w1_init'
        w2_tag = simTag+'_w2_init'
    else:
        w1_tag = simTag+'_w1_e'+str(epoch)
        w2_tag = simTag+'_w2_e'+str(epoch)

    param_tag = simTag+'_surrParams'
    try:
        params = getWeightFromUniqueId(param_tag, objFolder=objFolder)
        excClassActivity, inhClassActivity, excNumSpikes, inhNumSpikes, inpNumSpikes, isiByClass, distance = getActivityISIAndDistanceOverTime(w1_tag, w2_tag, params, objFolder, calcDistance)
    except Exception as e:
        print('ERROR CAUGHT:'+str(e))
        print('ERROR CONTEXT:'+str(e.__context__))
        writeHTMLNote(reportFile, note='ERROR LOADING DATA AND CALCULATING ACTIVITY '+str(e))
        writeHTMLSectionFooter(reportFile)
    
    # errors can arise if there was no activity in any particular class/group
    #  because of this, each section is wrapped in try/except blocks
    try:
        # Spike time table of distributions by output class
        writeHTMLNote(reportFile, note='Excitatory spikes per case')
        writeTableDistributionHeader(reportFile, htmlClass='dataframe excSpikesPerCase')
        for i in range(params.nb_outputs):
            writeDistributionRow(reportFile, simTag, excNumSpikes[i], notes='Exc data for Class '+str(i))
        writeTableDistributionFooter(reportFile)
    except Exception as e:
        print('ERROR CAUGHT:'+str(e))
        print(excNumSpikes)
        writeHTMLNote(reportFile, note='EXCITATORY DATA NOT SHOWN BECAUSE OF LACK OF ACTIVITY')
    try:
        writeHTMLNote(reportFile, note='Inhibitory spikes per case')
        writeTableDistributionHeader(reportFile, htmlClass='dataframe inhSpikesPerCase')
        for i in range(params.nb_outputs):
            writeDistributionRow(reportFile, simTag, inhNumSpikes[i], notes='Inh data for Class '+str(i))
        writeTableDistributionFooter(reportFile)
    except Exception as e:
        print('ERROR CAUGHT:'+str(e))
        print(inhNumSpikes)
        writeHTMLNote(reportFile, note='INHIBITORY DATA NOT SHOWN BECAUSE OF LACK OF ACTIVITY')

    # Interspike interval (ISI) table of distributions by output class
    try:
        writeHTMLNote(reportFile, note='InterSpike Interval Distributions')
        writeTableDistributionHeader(reportFile, htmlClass='dataframe isi')
        for i in range(params.nb_outputs):
            writeDistributionRow(reportFile, simTag, isiByClass[i], notes='ISI data for Class '+str(i))
        writeTableDistributionFooter(reportFile)
    except Exception as e:
        print('ERROR CAUGHT:'+str(e))
        print(isiByClass)
        writeHTMLNote(reportFile, note='INTERSPIKE INTERVAL DATA NOT SHOWN BECAUSE OF LACK OF ACTIVITY')
    
    # van rossum distance table
    #  for more details look for a paper titled: 'A Novel Spike Distance' by Van Rossum
    if calcDistance:
        writeHTMLNote(reportFile, note='Van Rossum Distance of Hidden Layer')
        writeTableDistributionHeader(reportFile, htmlClass='datafram vanRossum')
        writeDistributionRow(reportFile, simTag, distance)
        writeTableDistributionFooter(reportFile)
        writeDistanceMatrixTable(reportFile, distance)
    
    # spike times and ISI graphs
    for i in range(params.nb_outputs):
        descriptor = 'network_activity_case_'+str(i)+'_epoch_'+str(epoch)
        graphFile = generateSpikingActivity(simTag, descriptor, excClassActivity[i], inhClassActivity[i], ylabel='AVG Spikes Class '+str(i))
        writeImageHTML(reportFile, graphFile, simTag, descriptor)
        
        descriptor = 'network_num_exc_spikes_'+str(i)+'_epoch_'+str(epoch)
        graphFile = generateHistogram(simTag, descriptor, np.array(excNumSpikes[i]), xlabel='Num Exc Spikes', ylabel='Num Cases (Class '+str(i)+')')
        writeImageHTML(reportFile, graphFile, simTag, descriptor)

        descriptor = 'network_num_inh_spikes_'+str(i)+'_epoch_'+str(epoch)
        graphFile = generateHistogram(simTag, descriptor, np.array(excNumSpikes[i]), xlabel='Num Inh Spikes', ylabel='Num Cases (Class '+str(i)+')')
        writeImageHTML(reportFile, graphFile, simTag, descriptor)

        descriptor = 'network_ISI_case_'+str(i)+'_epoch_'+str(epoch)
        graphFile = generateISIDistribution(simTag, descriptor, isiByClass[i], ylabel='Total Frequency Class '+str(i), yscale='symlog')
        writeImageHTML(reportFile, graphFile, simTag, descriptor)

    writeHTMLSectionFooter(reportFile)

def writeAccSection(reportFile, simTagBase, trialRange, objFolder):
    # accuracy and loss section
    writeHTMLSectionHeader(reportFile, 'Accuracy and Loss Trials '+str(trialRange), note='Accuracy and Loss curves can be combined onto single plots for easy comparison')
    accAndTags = []
    lossAndTags = []
    for i in trialRange:
        simTag = simTagBase+'_'+str(i)
        accTag = simTag+'_accList'
        lossTag = simTag+'_lossList'
        accList = getWeightFromUniqueId(accTag, objFolder=objFolder)
        lossList = getWeightFromUniqueId(lossTag, objFolder=objFolder)

        accAndTags.append( (simTag,accList) )
        lossAndTags.append( (simTag,lossList) )

    writeAccTableDistributionHeader(reportFile)
    for accAndTag in accAndTags:
        (simTag,accList) = accAndTag
        writeAccDistributionRow(reportFile, simTag, np.array(accList))
    writeTableDistributionFooter(reportFile)

    accGraph, lossGraph = generateMultAccLossGraph(simTagBase, accAndTags, lossAndTags)
    writeImageHTML(reportFile, accGraph, simTag=simTag+'t'+str(min(trialRange))+'-t'+str(max(trialRange)),descriptor='accuracy')

    writeLossTableDistributionHeader(reportFile)
    for lossAndTag in lossAndTags:
        (simTag,lossList) = lossAndTag
        writeLossDistributionRow(reportFile, simTag, np.array(lossList))
    writeTableDistributionFooter(reportFile)
    writeImageHTML(reportFile, lossGraph, simTag=simTag+'t'+str(min(trialRange))+'-t'+str(max(trialRange)),descriptor='loss')
    writeHTMLSectionFooter(reportFile)

def writeSectionEIRatio(reportFile, objFolder, baseSimTag='sparse050_10class', trialRange=range(100,250), calcActivity=False, calcDistance=False, epoch=-1):
    # this is the main chunk of code for the report. We will look at a base simTag, and all the
    #  trials that were referenced in this. Often only a single trial is being analyzed, particularly
    #  if the activity and distances are being calculated, and so trialRange may be only a single value
    # regardless, we will be loading all the relevant weight, params, acc, and loss files to generate
    #  a complete picture of activity in the network

    writeHTMLSectionHeader(reportFile, baseSimTag)
    for i in trialRange:
        tempParam = getWeightFromUniqueId(baseSimTag+'_'+str(i)+'_surrParam', objFolder=objFolder)
        writeHTMLNote(reportFile, note=str(tempParam.__dict__))
    descriptor = '_weights'
    writeHTMLSectionHeader(reportFile, 'Sample Weight Dist', note='This is a test of weight distributionss\n')
    writeTableDistributionHeader(reportFile, htmlClass='dataframe weights')

    # sample weight distributions
    #  this analysis takes the initial weights and across each epoch and show the
    #  distribution of the weights using the writeDistributionRow of the generateHTML script
    # note that the graph files are also generated, but we will first build the entire
    #  table of distribution metrics, and then display all the graphs afterwards
    graphData = []
    for i in trialRange:
        try:
            # init training data
            simTag = baseSimTag+'_'+str(i)+'_w1_init'
            print(simTag)
            tmpW = getWeightFromUniqueId(simTag, objFolder=objFolder)

            writeDistributionRow(reportFile,simTag=simTag,notes='Input-Hidden W', rawValuesNP=tmpW.flatten().detach().numpy())
            graphFile = generateHistogram(simTag, 'testing', tmpW.flatten().detach().numpy(), xlabel='Weight', ylabel='Frequency', numBins=20, xmin=None, xmax=None, ymin=None, ymax=None)

            graphData.append([graphFile, simTag, descriptor])

            tmpP = getWeightFromUniqueId(baseSimTag+'_'+str(i)+'_surrParam', objFolder=objFolder)
            for e in range(tmpP.nb_epochs):
                simTag = baseSimTag+'_'+str(i)+'_w1_e'+str(e)
                print(simTag)
                tmpW = getWeightFromUniqueId(simTag, objFolder=objFolder)

                writeDistributionRow(reportFile,simTag=simTag,notes='', rawValuesNP=tmpW.flatten().detach().numpy())
                graphFile = generateHistogram(simTag, 'e'+str(e), tmpW.flatten().detach().numpy(), xlabel='Weight', ylabel='Frequency', numBins=20, xmin=None, xmax=None, ymin=None, ymax=None)

            print('HANDLED '+str(i))
        except:
            print('CAUGHT FAILED DATA '+str(i))

    writeTableDistributionFooter(reportFile)

    writeHTMLSectionHeader(reportFile, 'Weight Graphs', note='all weight distributions, note that the x axis and y axis are not the same across simulations')
    for graphDataSet in graphData:
        writeImageHTML(reportFile, graphDataSet[0], graphDataSet[1], graphDataSet[2])
    writeHTMLSectionFooter(reportFile)

    # Calculating activity
    #  this is an optional metric calculation, and is suggested when you are only analyzing a single trial
    #  because this can take a while to run
    if calcActivity:
        for i in trialRange:
            simTag = baseSimTag+'_'+str(i)
            writeActivitySection(reportFile, simTag, objFolder, calcDistance=calcDistance, epoch=epoch)

    # accuracy and loss section
    #  if multiple trials are listed in the trialRange, they will all show on a single plot
    writeAccSection(reportFile, baseSimTag, trialRange, objFolder)
    
    # end of E-I section
    writeHTMLSectionFooter(reportFile)

if __name__ == '__main__':
    # the script can be instantiated directly and run with various command line args
    #  check this script with python3.7 ./scripts/generateReport.py --help for more information

    parser = argparse.ArgumentParser(description='HTML Report Generator. Please ensure simulations have been run and data saved prior to running this script')
    parser.add_argument('-r', '--report_file', help='Ouptut HTML file path (ex. TEST-REPORT.html)', default='TEST-REPORT.html')
    parser.add_argument('-s', '--base_sim_tag', help='The base simTag. Note that \'_#\' will be attached to this string when building the particular simTags for the trials within range')
    parser.add_argument('-a', '--activity', help='Flag for whether to calculate activity, this drastically increases report generation time', action='store_true')
    parser.add_argument('-d', '--distance', help='Flag for whether to calcualte distance, this drastically increases report generation time', action='store_true')
    parser.add_argument('-t', '--trial_range', help='List the start (inclusive) and stopping index (exclusive) of trials to be processed', nargs=2, type=int)
    parser.add_argument('-e', '--epoch', help='Run particular epoch for the activity data', default=-1, type=int)
    parser.add_argument('-o', '--obj_path', help='Folder containing the pickled object files', default='./data/obj')

    args=parser.parse_args()
    if(args.distance and not args.activity):
        print('ERROR: CANNOT CALCULATE DISTANCE AND NOT ACTIVITY\nSEE --help FOR MORE INFORMATION')

    trialRange = range(args.trial_range[0], args.trial_range[1])
    writeHTMLHeader(args.report_file)
    writeSectionEIRatio(args.report_file, args.obj_path, args.base_sim_tag, trialRange=trialRange, calcActivity=args.activity, calcDistance=args.distance, epoch=args.epoch)
    writeHTMLFooter(args.report_file)
