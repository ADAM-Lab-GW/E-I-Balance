import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def writeHTMLHeader(filename, title=''):
    with open(filename, 'w') as f:
        f.write('<!DOCTYPE html>\n<html>\n<body>\n')
        f.write('<h1>'+title+'</h1>\n')

def writeHTMLFooter(filename):
    with open(filename, 'a') as f:
        f.write('</body>\n</html>')

def writeHTMLSectionHeader(filename, title, note=''):
    with open(filename, 'a') as f:
        f.write('<details>\n')
        f.write('<summary style="font-size:24px;">'+title+'</summary>\n')
        f.write('<p>'+note+'</p>\n')

def writeHTMLSectionFooter(filename):
    with open(filename, 'a') as f:
        f.write('<hr></hr>\n')
        f.write('</details>\n')

def writeHTMLNote(filename, note=''):
    with open(filename, 'a') as f:
        f.write('<p>'+note+'</p>\n')

def writePandasToHTML(filename, df):
    with open(filename, 'a') as f:
        f.write(df.to_html()+'\n')

def writeTableDistributionHeader(filename, htmlClass='dataframe'):
    with open(filename, 'a') as f:
        f.write('<table border="1" class="'+htmlClass+'">\n')
        f.write('<thead>\n')
        f.write('<tr style="text-align: center;">\n')
        f.write('<th>SimTag</th>\n')
        f.write('<th>Notes</th>\n')
        f.write('<th>Min</th>\n')
        f.write('<th>Q1</th>\n')
        f.write('<th>Median</th>\n')
        f.write('<th>Q3</th>\n')
        f.write('<th>Max</th>\n')
        f.write('<th>Mean</th>\n')
        f.write('<th>STD</th>\n')
        f.write('<th>Skew</th>\n')
        f.write('</tr>\n')
        f.write('</thead>\n')
        f.write('<tbody>\n')

def writeTableDistributionFooter(filename):
    with open(filename, 'a') as f:
        f.write('</tbody>\n')
        f.write('</table>\n')

def writeDistributionRow(filename, simTag, rawValuesNP, notes=''):
    # call the write
    minVal = np.min(rawValuesNP)
    q1 = np.quantile(rawValuesNP, 0.25) # first quartile
    median = np.quantile(rawValuesNP, 0.5) # median
    q3 = np.quantile(rawValuesNP, 0.75)
    maxVal = np.max(rawValuesNP)
    mean = np.mean(rawValuesNP)
    stdev = np.std(rawValuesNP)
    skew = 3*(mean-median)/stdev
    with open(filename, 'a') as f:
        f.write('<tr>\n')
        f.write('<td>{0}</td>\n'.format(simTag))
        f.write('<td>{0}</td>\n'.format(notes))
        f.write('<td>{0:.8f}</td>\n'.format(minVal))
        f.write('<td>{0:.8f}</td>\n'.format(q1))
        f.write('<td>{0:.8f}</td>\n'.format(median))
        f.write('<td>{0:.8f}</td>\n'.format(q3))
        f.write('<td>{0:.8f}</td>\n'.format(maxVal))
        f.write('<td>{0:.8f}</td>\n'.format(mean))
        f.write('<td>{0:.8f}</td>\n'.format(stdev))
        f.write('<td>{0:.8f}</td>\n'.format(skew))
        f.write('</tr>\n')

def writeAccTableDistributionHeader(filename):
    with open(filename, 'a') as f:
        f.write('<table border="1" class="dataframe">\n')
        f.write('<thead>\n')
        f.write('<tr style="text-align: center;">\n')
        f.write('<th>SimTag</th>\n')
        f.write('<th>Notes</th>\n')
        f.write('<th>Init Acc</th>\n')
        f.write('<th>Final Acc</th>\n')
        f.write('<th>Max Acc</th>\n')
        f.write('<th>Last 10epoch AVG</th>\n')
        f.write('<th>Last 10epoch STD</th>\n')
        f.write('<th>Epochs till AVG-STD</th>\n')
        f.write('</tr>\n')
        f.write('</thead>\n')
        f.write('<tbody>\n')

def writeAccDistributionRow(filename, simTag, rawAccNP, notes=''):
    # call the write
    initAccuracy = rawAccNP[0]
    finalAccuracy = rawAccNP[-1]
    maxAccuracy = np.max(rawAccNP)
    avgLast10Epochs = np.average(rawAccNP[-10:])
    stdLast10Epochs = np.std(rawAccNP[-10:])
    epochsTill1STD = np.argmax(rawAccNP>(avgLast10Epochs-stdLast10Epochs))-1 # -1 to handle the init

    with open(filename, 'a') as f:
        f.write('<tr>\n')
        f.write('<td>{0}</td>\n'.format(simTag))
        f.write('<td>{0}</td>\n'.format(notes))
        f.write('<td>{0:.8f}</td>\n'.format(initAccuracy))
        f.write('<td>{0:.8f}</td>\n'.format(finalAccuracy))
        f.write('<td>{0:.8f}</td>\n'.format(maxAccuracy))
        f.write('<td>{0:.8f}</td>\n'.format(avgLast10Epochs))
        f.write('<td>{0:.8f}</td>\n'.format(stdLast10Epochs))
        f.write('<td>{0}</td>\n'.format(epochsTill1STD))

def writeLossTableDistributionHeader(filename):
    with open(filename, 'a') as f:
        f.write('<table border="1" class="dataframe">\n')
        f.write('<thead>\n')
        f.write('<tr style="text-align: center;">\n')
        f.write('<th>SimTag</th>\n')
        f.write('<th>Notes</th>\n')
        f.write('<th>Init Loss</th>\n')
        f.write('<th>Final Loss</th>\n')
        f.write('<th>Max Loss</th>\n')
        f.write('<th>Last 10epoch AVG</th>\n')
        f.write('<th>Last 10epoch STD</th>\n')
        f.write('<th>Epochs till AVG+STD</th>\n')
        f.write('</tr>\n')
        f.write('</thead>\n')
        f.write('<tbody>\n')

def writeLossDistributionRow(filename, simTag, rawLossNP, notes=''):
    # call the write
    initLoss = rawLossNP[0]
    finalLoss = rawLossNP[-1]
    minLoss = np.min(rawLossNP)
    avgLast10Epochs = np.average(rawLossNP[-10:])
    stdLast10Epochs = np.std(rawLossNP[-10:])
    epochsTill1STD = np.argmax(rawLossNP<(avgLast10Epochs+stdLast10Epochs))-1 # -1 to handle the init

    with open(filename, 'a') as f:
        f.write('<tr>\n')
        f.write('<td>{0}</td>\n'.format(simTag))
        f.write('<td>{0}</td>\n'.format(notes))
        f.write('<td>{0:.8f}</td>\n'.format(initLoss))
        f.write('<td>{0:.8f}</td>\n'.format(finalLoss))
        f.write('<td>{0:.8f}</td>\n'.format(minLoss))
        f.write('<td>{0:.8f}</td>\n'.format(avgLast10Epochs))
        f.write('<td>{0:.8f}</td>\n'.format(stdLast10Epochs))
        f.write('<td>{0}</td>\n'.format(epochsTill1STD))

def writeDistanceMatrixTable(filename, distanceNP):
    with open(filename, 'a') as f:
        f.write('<table border="1" class="dataframe">\n')
        f.write('<thead>\n')
        f.write('<tr style="text-align: center;">\n')
        f.write('<th> </th>\n')
        for i in range(len(distanceNP)):
            f.write('<th>'+str(i)+'</th>\n')
        f.write('</tr>\n')
        f.write('</thead>\n')
        f.write('<tbody>\n')

        for i in range(len(distanceNP)):
            f.write('<tr>\n')
            f.write('<td>{0}</td>\n'.format(i))
            for j in range(len(distanceNP)):
                f.write('<td>{0:.8f}</td>\n'.format(distanceNP[i,j]))
            f.write('</tr>\n')

        f.write('</tbody>\n')
        f.write('</table>\n')

def generateHistogram(simTag, descriptor, rawValuesNP, xlabel, ylabel='Frequency', numBins=20, xmin=None, xmax=None, ymin=None, ymax=None):
    if xmin==None:
        xmin = np.min(rawValuesNP)
    if xmax==None:
        xmax = np.max(rawValuesNP)

    counts, bins = np.histogram(rawValuesNP,bins=numBins,range=(xmin,xmax))
    plt.clf()
    plt.grid(axis='both', which='both')
    plt.rc('axes',axisbelow=True)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.title(simTag)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([xmin,xmax])
    if(ymin!=None and ymax!=None):
        plt.ylim([ymin,ymax])

    graphFile = './reportPlots/'+simTag+'_'+descriptor
    plt.savefig(graphFile+'.png', dpi=600)
    plt.savefig(graphFile+'.svg')

    return graphFile+'.svg'

def generateSpikingActivity(simTag, descriptor, excActivity, inhActivity, xmin=0, xmax=100, ymin=0, ymax=100, ylabel='Number of Spikes'):
    plt.clf()
    plt.title(simTag)
    plt.xlabel('Time (ms)')
    plt.ylabel(ylabel)
    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])
    plt.grid(axis='both', which='both')
    plt.rc('axes',axisbelow=True)
    plt.bar(range(len(excActivity)), excActivity, width=1.0, label='Excitatory')
    plt.bar(range(len(inhActivity)), inhActivity, width=1.0, bottom=excActivity, label='Inhibitory')
    plt.legend()

    graphFile = './reportPlots/'+simTag+'_'+descriptor
    plt.savefig(graphFile+'.png', dpi=600)
    plt.savefig(graphFile+'.svg')

    return graphFile+'.svg'

def generateISIDistribution(simTag, descriptor, isiNP, xmin=None, xmax=None, ymin=None, ymax=None, ylabel='Frequency', yscale='linear'):
    plt.clf()
    plt.title(simTag)
    plt.xlabel('InterSpike Interval (ms)')
    plt.ylabel(ylabel)
    plt.yscale(yscale)
    plt.grid(axis='both', which='both')
    plt.rc('axes',axisbelow=True)
    if(xmin!=None and xmax!=None):
        plt.xlim([xmin,xmax])
    if(ymin!=None and ymax!=None):
        plt.ylim([ymin,ymax])

    values, counts = np.unique(isiNP, return_counts=True)
    plt.bar(values+1, counts, width=1.0)
    graphFile = './reportPlots/'+simTag+'_'+descriptor
    plt.savefig(graphFile+'.png', dpi=600)
    plt.savefig(graphFile+'.svg')

    return graphFile+'.svg'

def generateMultAccLossGraph(graphName, accAndTags, lossAndTags):
    # this allows for generating acc and loss curves that have
    #  multiple accuracy and loss curves onto a single graph
    #  each accAndTags, lossAndTags, are pairs of accuracy
    #  for example:
    #      accAndTags = [('test0', [0.1,0.7,0.8]), ('test1', [0.1,0.5,0.7])]
    
    # accuracy curves
    plt.clf()
    plt.title(graphName+' Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.ylim([0,100])
    plt.grid(axis='both', which='both')
    plt.rc('axes',axisbelow=True)
    for accTag in accAndTags:
        (simTag, accList) = accTag
        multiplier = 1.0
        if max(accList) < 1.0:
            # accList is currently a percentage between 0 and 1
            #  we need to scale it to be between 0 and 100
            multiplier = 100.0

        plt.plot(range(len(accList)), [a*multiplier for a in accList], label=simTag)

    plt.legend()
    accGraphFile = './reportPlots/'+graphName+'_accList'
    plt.savefig(accGraphFile+'.png', dpi=600)
    plt.savefig(accGraphFile+'.svg')

    # loss curves
    plt.clf()
    plt.title(graphName+' Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    for lossTag in lossAndTags:
        (simTag, lossList) = lossTag
        plt.plot(range(len(lossList)), lossList, label=simTag)

    plt.legend()
    lossGraphFile = './reportPlots/'+graphName+'_lossList'
    plt.savefig(lossGraphFile+'.png', dpi=600)
    plt.savefig(lossGraphFile+'.svg')

    return accGraphFile+'.svg', lossGraphFile+'.svg'

def writeImageHTML(filename, graphFile, simTag, descriptor):
    with open(filename, 'a') as f:
        f.write('<img src='+graphFile+' alt='+simTag+'_'+descriptor+' >\n')

if __name__ == "__main__":
    # sample output as example of how to use this scripts

    # setup the file
    filename='./tut_html.html'
    writeHTMLHeader(filename)

    # handling pandas DF
    df = pd.DataFrame(data={'col1': [1, 2], 'col2': [4, 3]})
    writeHTMLSectionHeader(filename, 'Sample Dataframe', note='This is a sample table generated from a dataframe')
    writePandasToHTML(filename, df)
    writeHTMLSectionFooter(filename)

    # handling numpy array and statistical breakdowns
    sampleNP = np.array([0,0,1,1,1,1,2,2,3,5,6])
    simTag = 'sampleNumpyArray'
    writeHTMLSectionHeader(filename, 'Sample distribution notes', note='This is a sample dataset of and the following data shown by statistics\n'+str(sampleNP)+'\n')
    writeTableDistributionHeader(filename)
    writeDistributionRow(filename,simTag=simTag,notes='This is a test row', rawValuesNP=sampleNP)
    writeTableDistributionFooter(filename)

    generateHistogram(simTag=simTag, descriptor='basic_distribution', rawValuesNP=sampleNP, xlabel='Values', ylabel='Frequency', numBins=7, xmin=0, xmax=7)

    writeHTMLSectionFooter(filename)

    writeHTMLSectionHeader(filename, 'Sample Layered sections', note='This is the outer section')
    writeHTMLSectionHeader(filename, 'Sample inner layer seciton', note='This is the inner section')
    writeHTMLSectionFooter(filename)
    writeHTMLSectionFooter(filename)

    # finish the file
    writeHTMLFooter(filename)
