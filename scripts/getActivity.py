from cmath import nan
import simLogger
import surrogateTraining
import weightAnalysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import scipy
from scipy.optimize import curve_fit
import matplotlib.font_manager as font_manager

def typeOuptut(output):
    numPositiveStepsCutoff = output.shape[1]/2

    # calculates the number of cases that fall into type 1 (both mainly positive output)
    output_max = np.max(output,axis=1) # max across time
    output_min = np.min(output,axis=1) # min across time

    bothPositive = np.count_nonzero((output_max[:,0]>0) & (output_max[:,1]>0))
    bothNegative = np.count_nonzero((output_max[:,0]==0) & (output_max[:,1]==0))
    singlePositive = np.count_nonzero(((output_max[:,0]>0)&(output_max[:,1]==0)) | ((output_max[:,0]==0)&(output_max[:,1]>0)))

    positiveVoltage = output>0
    tmp = np.count_nonzero(positiveVoltage, axis=1) # sum positive time steps
    
    positiveMajorityTime = np.count_nonzero(tmp>numPositiveStepsCutoff,axis=0) # count number of cases with majority time steps positive

    peakPos = np.count_nonzero(output_max>np.abs(output_min), axis=0)

    voltSTD = np.std(output,axis=1) # std across time
    voltSTD = np.average(voltSTD,axis=0)

    print('BOTH POS: '+str(bothPositive)+'\tBOTH NEG: '+str(bothNegative)+'\tSINGLE POS: '+str(singlePositive))
    return bothPositive, bothNegative, singlePositive, positiveMajorityTime, peakPos, voltSTD

def getActivityISIAndDistanceOverTime(w1_tag, w2_tag, params, obj_path, calcDistance):
    simLogger.logNotes('LOADING WEIGHTS')
    w1 = weightAnalysis.getWeightFromUniqueId(w1_tag, objFolder=obj_path)
    w2 = weightAnalysis.getWeightFromUniqueId(w2_tag, objFolder=obj_path)

    simLogger.logNotes('LOADING DATASET')
    x_train, x_test, y_train, y_test = surrogateTraining.getDataSet(params)
    
    simLogger.logNotes('SETTING RANGE OF CLASS DATA')

    # activity over time variables
    simLogger.logNotes('SETTING UP VARS FOR ACTIVITY')
    allActivityOverTime = np.array([])
    excActivityOverTime = np.array([])
    inhActivityOverTime = np.array([])

    excNumSpikes = [ [] for _ in range(params.nb_outputs) ]
    inhNumSpikes = [ [] for _ in range(params.nb_outputs) ]
    inpNumSpikes = [ [] for _ in range(params.nb_outputs) ]

    excClassActivity = np.array([[0.0]*params.nb_steps]*params.nb_outputs)
    inhClassActivity = np.array([[0.0]*params.nb_steps]*params.nb_outputs)
    numClasses = np.array([0]*params.nb_outputs)
    batch = 0

    # ISI calculation variable
    isiByClass = [np.array([])]*params.nb_outputs

    # van rossum distance variables
    if calcDistance:
        simLogger.logNotes('SETTING UP VARS FOR VAN ROSSUM')
    decay = np.zeros( (params.batch_size, params.nb_hidden) )
    decayFactor = 0.5
    distance = np.zeros( (params.nb_hidden,params.nb_hidden) )
    
    exp_decay = []
    tau = 1
    def_e = 2.718281828459
    for t in range(params.nb_steps):
        exp_decay.append(def_e ** (-1*t/tau))

    simLogger.logNotes('RUNNING TRAINING CASES')
    for x_local, y_local, batch_index in surrogateTraining.sparse_data_generator_preloaded(x_train, y_train, params):
        simLogger.logNotes('BATCH: '+str(batch))
        output, other_recordings = surrogateTraining.run_snn(x_local.to_dense(), w1, w2, params)
        mem_rec, spk_rec = other_recordings
        output_np = output.detach().numpy()
        spk_rec_np = spk_rec.detach().numpy()
        simLogger.logNotes('RAN BATCH DATA')

        allActivityOverTime = np.sum(spk_rec_np, axis=2)
        excActivityOverTime = np.sum(spk_rec_np[:,:,0:int(params.nb_hidden*params.excHidRat)], axis=2)
        inhActivityOverTime = np.sum(spk_rec_np[:,:,int(params.nb_hidden*params.excHidRat):int(params.nb_hidden)], axis=2)

        simLogger.logNotes('AGGREGATE ACTIVITY')
        for i in range(params.nb_outputs):
            class_batch_index = (y_local == i)
            numClasses[i] += np.count_nonzero(class_batch_index)
            exc = spk_rec_np[class_batch_index,:,0:int(params.nb_hidden*params.excHidRat)]
            inh = spk_rec_np[class_batch_index,:,int(params.nb_hidden*params.excHidRat):int(params.nb_hidden)]
            excClassActivity[i,:] = np.add(excClassActivity[i,:], np.sum(np.sum(exc, axis=2),axis=0)) # sum across neuron
            inhClassActivity[i,:] = np.add(inhClassActivity[i,:], np.sum(np.sum(inh, axis=2),axis=0)) #  and then cases 
            
            excNumSpikes[i].extend(np.sum(np.sum(exc, axis=2), axis=1).tolist()) # sum across neuron
            inhNumSpikes[i].extend(np.sum(np.sum(inh, axis=2), axis=1).tolist()) #  and then time

        simLogger.logNotes('LOG NUM INPUT SPIKES')
        for i in range(len(x_local)):
            inpNumSpikes[y_local[i].item()].append(x_local[i].coalesce().indices().shape[1])
    
        simLogger.logNotes('ISI CALCULATIONS')
        for i in range(len(spk_rec_np)):
            # generate ISI intervals from each case
            spk = spk_rec_np[i]
            isi = np.diff(np.where(spk==1)[0])
            if(len(isi)>0):
                isiByClass[y_local[i]] = np.append(isiByClass[y_local[i]], isi)

        simLogger.logNotes('DISTANCE CALCULATIONS')
        if calcDistance:
            for i in range(params.batch_size):
                batch_spk = spk_rec_np[t,:,:]
                decays = np.zeros(batch_spk.shape)
                # each spike train convolved with the exponential decay
                for n in range(params.nb_hidden):
                    decays[:,n] = np.convolve(batch_spk[:,n],exp_decay)[:params.nb_steps]
                    for m in range(n):
                        tmp = np.sqrt(np.sum(np.square(decays[:,n]-decays[:,m]))/tau)
                        distance[n,m] = distance[n,m]+tmp
                        distance[m,n] = distance[m,n]+tmp
                    
            # ---------INCORRECT VAN ROSSUM CALC
            #            for t in range(params.nb_steps):
            #                new_decay = decay*decayFactor + spk_rec_np[:,t,:]
            #                for i in range(params.nb_hidden):
            #                    for j in range(i+1):
            #                        dist = np.sum(np.square(new_decay[:,i]-new_decay[:,j]))
            #                        distance[i,j] = distance[i,j]+dist
            #                        distance[j,i] = distance[j,i]+dist
            #              decay = new_decay

        batch += 1

    simLogger.logNotes('FINISHED SIMULATIONS')
    # average the activity
    for i in range(params.nb_outputs):
        excClassActivity[i,:] = np.divide(excClassActivity[i,:],numClasses[i])
        inhClassActivity[i,:] = np.divide(inhClassActivity[i,:],numClasses[i])

    # finish van rossum distance calculation
    if calcDistance:
        distance = distance/params.batch_size
        return excClassActivity, inhClassActivity, excNumSpikes, inhNumSpikes, inpNumSpikes, isiByClass, distance
    
    # if not calculating distance, return None for distance
    return excClassActivity, inhClassActivity, excNumSpikes, inhNumSpikes, inpNumSpikes, isiByClass, None


def getTotalActivity(w1_tag, w2_tag, params, obj_path):
    w1 = weightAnalysis.getWeightFromUniqueId(w1_tag, objFolder=obj_path)
    w2 = weightAnalysis.getWeightFromUniqueId(w2_tag, objFolder=obj_path)

    x_train, x_test, y_train, y_test = surrogateTraining.getDataSet()
    train_idx = (y_train < params.nb_outputs)
    test_idx = (y_test < params.nb_outputs)
    
    x_train = x_train[train_idx]
    x_test = x_test[test_idx]
    y_train = y_train[train_idx]
    y_test = y_test[test_idx]

    total_active = 0
    total_active_exc = 0
    total_active_inh = 0
    total_cases = 0
    active_np = np.array([])
    batch_num = 0
    output_avg_lst = []
    output_min_lst = []
    output_max_lst = []
    both_positive_lst = []
    both_negative_lst = []
    single_positive_lst = []
    positive_time_class_0_lst = []
    positive_time_class_1_lst = []
    peak_pos_class_0_lst = []
    peak_pos_class_1_lst = []
    volt_std_class_0_lst = []
    volt_std_class_1_lst = []
    for x_local, y_local, batch_index in surrogateTraining.sparse_data_generator(x_train, y_train, params):
        output, other_recordings = surrogateTraining.run_snn(x_local.to_dense(), w1, w2, params)
        mem_rec, spk_rec = other_recordings
        output_np = output.detach().numpy()
        spk_rec_np = spk_rec.detach().numpy()
        bothPositive, bothNegative, singlePositive, positiveMajorityTime, peakPos, voltSTD = typeOuptut(output_np)
        both_positive_lst.append(bothPositive)
        both_negative_lst.append(bothNegative)
        single_positive_lst.append(singlePositive)
        positive_time_class_0_lst.append(positiveMajorityTime[0])
        positive_time_class_1_lst.append(positiveMajorityTime[1])
        peak_pos_class_0_lst.append(peakPos[0])
        peak_pos_class_1_lst.append(peakPos[1])
        volt_std_class_0_lst.append(voltSTD[0])
        volt_std_class_1_lst.append(voltSTD[1])
        output_avg = np.average(output_np,axis=0) # avg across cases
        output_min = np.min(output_np,axis=1) # min across time
        output_min = np.average(output_min, axis=0) # average across cases
        output_max = np.max(output_np,axis=1) # max across time
        output_max = np.average(output_max,axis=0) # average across cases
        #print("mem_rec_avg: "+str(output_avg.shape))
        #print("mem_rec_min: "+str(output_max.shape))
        output_avg_lst.append(output_avg)
        output_min_lst.append(output_min)
        output_max_lst.append(output_max)
        num_active = np.count_nonzero(np.sum(spk_rec_np,axis=1), axis=1)
        active_np = np.concatenate((active_np,num_active))
        total_active += np.sum(spk_rec_np)
        total_active_exc += np.sum(spk_rec_np[:,:,0:int(params.nb_hidden*params.excHidRat)])
        total_active_inh += np.sum(spk_rec_np[:,:,int(params.nb_hidden*params.excHidRat):int(params.nb_hidden)])
        total_cases += spk_rec_np.shape[0]
        print('BATCH: '+str(batch_num)+'\t SPIKES: '+str(np.sum(num_active))+'\t CASES HANDLED:'+str(total_cases))
        
        if batch_num==0:
            surrogateTraining.plot_voltage_traces(output,suptitle=params.simTag,fileName='./sparseTestVoltage/'+params.simTag+'.png', y_data=y_train, batch_indices=batch_index, dim=(6,6))

        batch_num+=1
    
    totalBothPositive = np.sum(np.array(both_positive_lst))
    totalBothNegative = np.sum(np.array(both_negative_lst))
    totalSinglePositive = np.sum(np.array(single_positive_lst))
    totalPositiveTimeClass0 = np.sum(np.array(positive_time_class_0_lst))
    totalPositiveTimeClass1 = np.sum(np.array(positive_time_class_1_lst))
    totalPeakPosClass0 = np.sum(np.array(peak_pos_class_0_lst))
    totalPeakPosClass1 = np.sum(np.array(peak_pos_class_1_lst))
    avgSTDClass0 = np.average(np.array(volt_std_class_0_lst))
    avgSTDClass1 = np.average(np.array(volt_std_class_1_lst))
    output_avg = np.average(np.array(output_avg_lst),axis=0) # final output average is 2d, time and neuron
    output_min_avg = np.average(np.array(output_min_lst),axis=0) # final min is 1d, neuron
    output_max_avg = np.average(np.array(output_max_lst),axis=0) # final max is 1d, neuron
    print("OUTPUT VOLTAGE SHAPE: "+str(output_avg.shape))
    print('TOTAL: '+str(total_active))
    return total_active, total_active_exc, total_active_inh, active_np, output_avg, output_min_avg, output_max_avg, totalBothPositive, totalBothNegative, totalSinglePositive, totalPositiveTimeClass0, totalPositiveTimeClass1, totalPeakPosClass0, totalPeakPosClass1, avgSTDClass0, avgSTDClass1

def writeSparseTestActivityHeader(outputFile='./sparseTestActivity.csv'):
    with open(outputFile,'a') as f:
        f.write('simTag,sparsity_w1,std_w1,total_active,total_active_exc,total_active_inh,class_0_avg,class_0_min,class_0_max,class_1_avg,class_1_min,class_1_max,totalBothPositive,totalBothNegative,totalSinglePositive,totalPositiveTimeClass0,totalPositiveTimeClass1,totalPeakPosClass0,totalPeakPosClass1,avgSTDClass0,avgSTDClass1\n')

def getSparseTestActvity(w1_tag,w2_tag,param_tag, idTag = None, outputFile='./sparseTestActivity.csv',  obj_path='/mnt/l/Grad School/Research/Repos/data_archive_2023_06/sparseTest'):
    print("Loading data files...")
    params=None
    try:
        params = weightAnalysis.getWeightFromUniqueId(param_tag, objFolder=obj_path)
    except:
        print('NO PARAMS FOUND: '+param_tag)
        return

    if idTag==None:
        idTag=params.simTag

    print('Collecting activity...')
    total_active, total_active_exc, total_active_inh, active_np, output_avg, output_min_avg, output_max_avg, totalBothPositive, totalBothNegative, totalSinglePositive, totalPositiveTimeClass0, totalPositiveTimeClass1,totalPeakPosClass0, totalPeakPosClass1, avgSTDClass0, avgSTDClass1 = getTotalActivity(w1_tag, w2_tag, params, obj_path)
    simLogger.saveObj(params.simTag,'num_active',active_np,makeNote=True)
    with open(outputFile,'a') as f:
        outStr = idTag+','
        outStr+= str(params.sparsity_w1)+','+str(params.std_w1)+','
        outStr+= str(total_active)+','+str(total_active_exc)+','+str(total_active_inh)+','
        outStr+= str(np.average(output_avg[:,0]))+','+str(output_min_avg[0])+','+str(output_max_avg[0])+','
        outStr+= str(np.average(output_avg[:,1]))+','+str(output_min_avg[1])+','+str(output_max_avg[1])+','
        outStr+= str(totalBothPositive)+','+str(totalBothNegative)+','+str(totalSinglePositive)+','
        outStr+= str(totalPositiveTimeClass0)+','+str(totalPositiveTimeClass1) + ','
        outStr+= str(totalPeakPosClass0)+','+str(totalPeakPosClass1)+','
        outStr+= str(avgSTDClass0)+','+str(avgSTDClass1)
        f.write(outStr+'\n')

    print('COMPLETED: '+params.simTag)

def generateActivityHistogram(id,trial):
    active_tag = 'sparseTest_'+str(id)+'_t'+str(trial)+'_num_active'
    active = weightAnalysis.getWeightFromUniqueId(active_tag, objFolder='./data/obj')
    print(active)
    print(np.min(active))
    print(np.max(active))
    counts, bins = np.histogram(active,bins=20,range=(0,100))
    plt.clf()
    plt.hist(bins[:-1], bins, weights=counts)
    plt.title(active_tag)
    plt.xlabel('Number Active')
    plt.ylabel('Case Frequency')
    plt.xlim([0,100])
    plt.ylim([0,12000])
    plt.yscale('symlog')
    plt.savefig('./sparsePlots/'+active_tag+'.png')

def getColors(row):
    #if row['learnRate'] == 0.001: return 'red'
    #if row['learnRate'] == 0.0001: return 'blue'
    if row['sparsity_w1'] == 0: return 'darkorange'
    if row['sparsity_w1'] == 0.25: return 'green'
    if row['sparsity_w1'] == 0.75: return 'purple'
    #if row['excHidRat'] == 0.5: return 'red'
    #if row['excHidRat'] == 0.75: return 'blue'
    print('error '+str(row['sparsity_w1']))
    print('simTag ' + str(row['simTag']))

def fitFunc(x, a, b, c):
    return a/(x-b) + c

def combineActivityAcc(spikeType='total_active', combinedLog='./data/logs/sparseTest_combined_trimmed2.csv', activityData='./sparseTestActivityClassifiedOutput2.csv', plotFileBase='./AAAI/activity'):
    print("Loading data...")
    df1 = pd.read_csv(combinedLog)
    print(df1.head())
    df2 = pd.read_csv(activityData)
    print(df2.head())
    df1['color'] = df1.apply(lambda row: getColors(row), axis=1)

    simTags = df2["simTag"].values.tolist()
    print(simTags)

    print("Generating coordinates...")
    x = []
    y = []
    c = []
    for simTag in simTags:
        if df2[df2['simTag']==simTag][spikeType].values[0] == 0:
            # we are going to skip plotting poiints at no spiking
            continue

        # here is the formula for converting to frequency from spikes
        #  there are 11776 trials counted, and each is 0.1sec
        freq_tmp = df2[df2['simTag']==simTag][spikeType].values[0] / 11776 / 0.1 / 100
        x.append(freq_tmp)
        y.append(df1[df1['simTag']==simTag]['maxAcc'].values[0])
        c.append(df1[df1['simTag']==simTag]['color'].values[0])

    print("Create Plot...")
    hfont = {'fontname':'Arial'}
    plt.clf()
    fig, axs = plt.subplots(1,1)
    plt.scatter(x,y,color=c,alpha=0.2)
    #plt.xlabel('Total Dataset Hidden Layer Spikes')
    plt.xlabel('Average Hidden Layer Neuron Firing Rate')
    plt.ylabel('Accuracy',**hfont)
    plt.xscale('symlog')
    #plt.yscale('log')
    plt.ylim([0.0,1])
    #plt.xlim([0,1.75e7])
    plt.xlim([0,1.75e7/11776/0.1/100])
    custom_lines = [Line2D([0], [0], marker='o', color='w', label='0.0',
                           markerfacecolor='darkorange', markersize=15),
                Line2D([0], [0], marker='o', color='w', label='0.25',
                           markerfacecolor='green', markersize=15),
                Line2D([0], [0], marker='o', color='w', label='0.75',
                           markerfacecolor='purple', markersize=15)]

    font = font_manager.FontProperties(family='Arial', style='normal', size=16)

    axs.legend(custom_lines,['0.0','0.25','0.75'],title='Input-Hidden Initial Sparsity', loc=3, prop=font)
    axs.set_box_aspect(1)
    plt.grid(which='major')
    plt.minorticks_on()
    plt.grid(which='minor')
    plt.savefig(plotFileBase+'_'+spikeType+'_Acc.png',dpi=600)
    plt.savefig(plotFileBase+'_'+spikeType+'_Acc.svg')
    print("Plot Saved")

    print("Create Plot...")
    plt.clf()
    fig, axs = plt.subplots(1,1)
    plt.scatter(x,y,color=c,alpha=0.2)
    #plt.xlabel('Total Dataset Hidden Layer Spikes')
    plt.xlabel('Average Hidden Layer Neuron Firing Rate')
    plt.ylabel('Accuracy')
    plt.xscale('symlog')
    #plt.yscale('log')
    plt.ylim([0.95,1])
    #plt.xlim([0,1.75e7])
    plt.xlim([0,1.75e7/11776/0.1/100])
    axs.legend(custom_lines,['0.0','0.25','0.75'],title='Input-Hidden Initial Sparsity', loc=3)
    axs.set_box_aspect(1)
    plt.grid(which='major')
    plt.minorticks_on()
    plt.grid(which='minor')
    plt.savefig(plotFileBase+'_'+spikeType+'_Acc_zoom.png',dpi=600)
    plt.savefig(plotFileBase+'_'+spikeType+'_Acc_zoom.svg')
    print("Plot Saved")

    x_clean = [[],[],[]]
    y_clean = [[],[],[]]
    colors = ['darkorange','green','purple']
    for simTag in simTags:
        x_tmp = df2[df2['simTag']==simTag][spikeType].values[0]
        y_tmp = df1[df1['simTag']==simTag]['maxAcc'].values[0]
        #if df1[df1['simTag']==simTag]['maxAcc'].values[0]<0.95:
        #    continue
        if x_tmp == 0:
            # remove points of no activity
            continue
        for i in range(len(colors)):
            if df1[df1['simTag']==simTag]['color'].values[0]==colors[i]:
                x_clean[i].append(x_tmp)
                y_clean[i].append(y_tmp)
                break

    print("Create Plot...")
    plt.clf()
    fig, axs = plt.subplots(1,1)
    plt.scatter(x,y,color=c,alpha=0.3)
    plt.xlabel('Total Dataset Hidden Layer Spikes')
    plt.ylabel('Accuracy')
    #plt.xscale('symlog')
    #plt.yscale('log')
    plt.ylim([0.0,1])
    plt.xlim([0,1.75e7])
    axs.legend(custom_lines,['0.0','0.25','0.75'],title='Input-Hidden Initial Sparsity', loc=3)
    axs.set_box_aspect(1)

    slope = []
    intercept = []
    r_value = []
    p_value = []
    std_err = []
    for i in range(len(colors)):
        slope_tmp, intercept_tmp, r_value_tmp, p_value_tmp, std_err_tmp = scipy.stats.linregress(x_clean[i], y_clean[i])
        slope.append(slope_tmp)
        intercept.append(intercept_tmp)
        r_value.append(r_value_tmp)
        p_value.append(p_value_tmp)
        std_err.append(std_err_tmp)
        #plt.plot([0,1e7],[intercept_tmp, slope_tmp*1e7+intercept_tmp], color=colors[i])
        #plt.text(0.4e7,0.997-(i/300), ('r2=%s'% float('%.3g'%r_value_tmp**2)),color=colors[i],size=10)
        
        #popt, pcov = curve_fit(fitFunc, x_clean[i], y_clean[i], bounds=([-0.000000000000001,-5,-1],[-0.000000000000001,5,2]))
        #  func y=ax^2+bx+c

        popt, pcov = curve_fit(fitFunc, x_clean[i], y_clean[i], bounds=([500000,100000,0],[10000000,100000000,2]))
        print(popt)
        print(pcov)
        x_exp = np.linspace(0,1.75e7,num=100)
        y_exp = fitFunc(x_exp,popt[0],popt[1],popt[2])
        plt.plot(x_exp,y_exp, color=colors[i])

    plt.savefig(plotFileBase+'_'+spikeType+'_Acc_bestFit.png',dpi=600)
    plt.savefig(plotFileBase+'_'+spikeType+'_Acc_bestFit.svg')

    plt.ylim([0.0,1])
    plt.savefig(plotFileBase+'_'+spikeType+'_Acc_zoom_bestFit.png',dpi=600)
    plt.savefig(plotFileBase+'_'+spikeType+'_Acc_zoom_bestFit.svg')

    print("Plot Saved")

def combineActivityExcInhAcc(combinedLog='./data/logs/sparseTest_combined_trimmed2.csv', activityData='./sparseTestActivityClassifiedOutput2.csv', plotFileBase='./AAAI/accuracy_type'):
    print("Loading data...")
    df1 = pd.read_csv(combinedLog)
    df2 = pd.read_csv(activityData)
    df1['color'] = df1.apply(lambda row: getColors(row), axis=1)
    print(df1.head())
    print(df2.head())

    simTags = df2["simTag"].values.tolist()
    print(simTags)

    print("Generating coordinates...")
    x = []
    y = []
    c = []
    s = []
    for simTag in simTags:
        total = df2[df2['simTag']==simTag]['total_active'].values[0]
        maxAcc = df1[df1['simTag']==simTag]['maxAcc'].values[0]
        std_w1 = df2[df2['simTag']==simTag]['std_w1'].values[0]
        sparsity_w1 = df2[df2['simTag']==simTag]['sparsity_w1'].values[0]
        diff = df2[df2['simTag']==simTag]['total_active_exc'].values[0]-df2[df2['simTag']==simTag]['total_active_inh'].values[0]
        exc = df2[df2['simTag']==simTag]['total_active_exc'].values[0]
        inh = df2[df2['simTag']==simTag]['total_active_inh'].values[0]
        class0avg = df2[df2['simTag']==simTag]['class_0_avg'].values[0]
        class1avg = df2[df2['simTag']==simTag]['class_1_avg'].values[0]
        class0min = df2[df2['simTag']==simTag]['class_0_min'].values[0]
        class1min = df2[df2['simTag']==simTag]['class_1_min'].values[0]
        class0max = df2[df2['simTag']==simTag]['class_0_max'].values[0]
        class1max = df2[df2['simTag']==simTag]['class_1_max'].values[0]
        bothPositive = df2[df2['simTag']==simTag]['totalBothPositive'].values[0]
        bothNegative = df2[df2['simTag']==simTag]['totalBothNegative'].values[0]
        singlePositive = df2[df2['simTag']==simTag]['totalSinglePositive'].values[0]
        totalPositiveTimeClass0 = df2[df2['simTag']==simTag]['totalPositiveTimeClass0'].values[0]
        totalPositiveTimeClass1 = df2[df2['simTag']==simTag]['totalPositiveTimeClass1'].values[0]
        totalPeakPosClass0 = df2[df2['simTag']==simTag]['totalPeakPosClass0'].values[0]
        totalPeakPosClass1 = df2[df2['simTag']==simTag]['totalPeakPosClass1'].values[0]
        print(simTag+','+str(maxAcc)+','+str(total)+','+str(diff)+','+str(exc)+','+str(inh)+','+str(class0avg)+','+str(class1avg)+','+str(class0min)+','+str(class1min)+','+str(class0max)+','+str(class1max)+','+str(bothPositive)+','+str(bothNegative)+','+str(singlePositive)+','+str(totalPositiveTimeClass0)+','+str(totalPositiveTimeClass1)+','+str(totalPeakPosClass0)+','+str(totalPeakPosClass0))
        # w1 = weightAnalysis.getWeightFromUniqueId(simTag+'_w1_init',objFolder='/mnt/l/Grad School/Research/Repos/data_archive_2023_06/sparseTest').detach().numpy()
        # w2 = weightAnalysis.getWeightFromUniqueId(simTag+'_w2_init',objFolder='/mnt/l/Grad School/Research/Repos/data_archive_2023_06/sparseTest').detach().numpy()
        # combined_weights = np.dot(w1,w2)
        # class0AvgWeighting = np.sum(combined_weights[:,0])
        # class1AvgWeighting = np.sum(combined_weights[:,1])

        # if maxAcc>0.6:
        #     continue
        # if diff>0:
        #     c.append('tab:orange')
        # else:
        #     c.append('tab:green')

        y.append(maxAcc)
        x.append(total)
        c.append(sparsity_w1)
        s.append(20)
        #s.append((maxAcc-0.4)*100)
        
    print("Create Plot...")
    plt.clf()
    fig, axs = plt.subplots(1,1)
    scatter = plt.scatter(x,y,c=c,s=s,alpha=0.5)
    colorbar = plt.colorbar(scatter)
    colorbar.set_label('Cases with both Negative Output')

    plt.xlabel('Total Hidden Layer Spikes')
    plt.ylabel('Accuracy')
    #plt.xscale('symlog')
    #plt.yscale('symlog')
    #plt.ylim([0,1])
    #plt.xlim([0,1])
    axs.legend('Cases with Both Output Negative')
    #axs.set_box_aspect(1)
    plt.tight_layout()
    plt.savefig(plotFileBase+'_output.png',dpi=600)
    plt.savefig(plotFileBase+'_output.svg')
    print("Plot Saved")

def combineActivityLoss():
    print("Loading data...")
    df1 = pd.read_csv('./data/logs/sparseTest_combined_trimmed2.csv')
    df2 = pd.read_csv('./sparseTestActivity.csv')
    df1['color'] = df1.apply(lambda row: getColors(row), axis=1)

    simTags = df2["simTag"].values.tolist()
    print(simTags)

    print("Generating coordinates...")
    x = []
    y = []
    c = []
    for simTag in simTags:
        x.append(df2[df2['simTag']==simTag]['totalSpikes'].values[0])
        y.append(df1[df1['simTag']==simTag]['minLoss'].values[0])
        c.append(df1[df1['simTag']==simTag]['color'].values[0])

    print("Create Plot...")
    plt.clf()
    fig, axs = plt.subplots(1,1)
    plt.scatter(x,y,color=c,alpha=0.3)
    plt.xlabel('Total Dataset Hidden Layer Spikes')
    plt.ylabel('Loss')
    #plt.xscale('symlog')
    plt.ylim([0,1])
    custom_lines = [Line2D([0], [0], marker='o', color='w', label='0.0',
                           markerfacecolor='darkorange', markersize=15),
                Line2D([0], [0], marker='o', color='w', label='0.25',
                           markerfacecolor='green', markersize=15),
                Line2D([0], [0], marker='o', color='w', label='0.75',
                           markerfacecolor='purple', markersize=15)]
    axs.legend(custom_lines,['0.0','0.25','0.75'],title='Input-Hidden Initial Sparsity', loc=3)
    axs.set_box_aspect(1)
    plt.savefig('./AAAI/activityLoss.png', dpi=600)
    plt.savefig('./AAAI/activityAcc.svg')
    print("Plot Saved")

def weightAccCor(logFile='./data/logs/sparseExcHid/fullSparseExcHid.csv', obj_path='./sparseExcHid95/sparseExcHid', plotFileBase='./sparseExcHid95/weight'):
    df1 = pd.read_csv(logFile)
    df1['color'] = df1.apply(lambda row: getColors(row), axis=1)
    simTags = df1["simTag"].values.tolist()
    w1=[]
    w2=[]
    acc=[]
    for simTag in simTags:
        print('LOADING '+simTag)
        w1_tag = df1[df1['simTag']==simTag]['w1_init'].values[0].split('/')[-1]
        w2_tag = df1[df1['simTag']==simTag]['w2_init'].values[0].split('/')[-1]
        tmp = weightAnalysis.getWeightFromUniqueId(w1_tag, objFolder=obj_path).detach().numpy()
        print(tmp[0,0])
        if(tmp[0,0]==nan):
            print("ERROR")
            return
        w1.append(tmp)
        tmp = weightAnalysis.getWeightFromUniqueId(w2_tag, objFolder=obj_path).detach().numpy()
        print(tmp[0,0])
        if(tmp[0,0]==nan):
            print("ERROR")
            return
        w2.append(tmp)
        acc.append(df1[df1['simTag']==simTag]['maxAcc'].values[0])
        print(type(df1[df1['simTag']==simTag]['maxAcc'].values[0]))
        if(type(df1[df1['simTag']==simTag]['maxAcc'].values[0])==np.nan):
            print("ERROR")
            return
    
    w1 = np.array(w1)
    w2 = np.array(w2)
    acc = np.array(acc)

    cor = np.zeros((28,28))
    for i in range(28):
        for j in range(28):
            cor[i,j] = np.sum(w1[:,i,j])

            print(str(i)+','+str(j)+'='+str(cor[i,j]))

    plt.imshow(cor, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.savefig(plotFileBase+'Cor.png', dpi=600)
    plt.savefig(plotFileBase+'Cor.svg')

    plt.clf()
    plt.hist(cor.flatten(), bins='auto')
    plt.ylabel('frequency')
    plt.xlabel('correlation between weight and acc')
    plt.savefig(plotFileBase+'eCorHist.png', dpi=600)
    plt.savefig(plotFileBase+'CorHist.svg')

    cor = np.zeros((100,2))
    for i in range(100):
        for j in range(2):
            cor[i,j] = np.correlate(w2[:,i,j],acc)

            print(str(i)+','+str(j)+'='+str(cor[i,j]))

    plt.clf()
    plt.imshow(cor, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.savefig(plotFileBase+'Cor2.png', dpi=600)
    plt.savefig(plotFileBase+'Cor2.svg')

    plt.clf()
    plt.hist(cor.flatten(), bins='auto')
    plt.ylabel('frequency')
    plt.xlabel('correlation between weight and acc')
    plt.savefig(plotFileBase+'CorHist2.png', dpi=600)
    plt.savefig(plotFileBase+'CorHist2.svg')

def calcAccStats(combinedLog='./data/logs/sparseTest_combined_trimmed2.csv', activityData='./sparseTestActivityClassifiedOutput2.csv', plotFileBase='./sparseExcHid95/weight', title='Default'):
    print("Loading data...")
    df1 = pd.read_csv(combinedLog)
    df2 = pd.read_csv(activityData)
    df1['color'] = df1.apply(lambda row: getColors(row), axis=1)
    print(df1.head())
    print(df2.head())

    # aggregate data by std and sparsity trials
    simTags = df1["simTag"].values.tolist()
    accDict = {}
    for simTag in simTags:
        print('LOADING '+simTag)
        acc_tmp = df1[df1['simTag']==simTag]['maxAcc'].values[0]
        sparse_tmp = df1[df1['simTag']==simTag]['sparsity_w1'].values[0]
        std_tmp = df1[df1['simTag']==simTag]['std_w1'].values[0]
        key_tmp = (sparse_tmp,std_tmp)

        if key_tmp in accDict.keys():
            accDict[key_tmp].append(acc_tmp)
        else:
            accDict[key_tmp] = [acc_tmp]
        
    # calc average and std for trials
    avgDict = {}
    stdDict = {}
    print('sparseW\tstdW\tavgA\tstdA')
    for key in accDict.keys():
        avgDict[key] = np.average(np.array(accDict[key]))
        stdDict[key] = np.std(np.array(accDict[key]))
        (sparse_tmp,std_tmp) = key
        print(str(sparse_tmp)+'\t'+str(std_tmp)+'\t'+str(avgDict[key])+'\t'+str(stdDict[key]))


    for cutoff in [0.001, 0.01, 0.02, 0.03, 0.05]:
        x = []
        y = []
        c = []
        total_bad_trials = 0
        total_trials = 0
        total_no_activity = 0
        for simTag in simTags:
            acc_tmp = df1[df1['simTag']==simTag]['maxAcc'].values[0]
            sparse_tmp = df1[df1['simTag']==simTag]['sparsity_w1'].values[0]
            std_tmp = df1[df1['simTag']==simTag]['std_w1'].values[0]
            activity_tmp = df2[df2['simTag']==simTag]['total_active'].values[0]
            key_tmp = (sparse_tmp,std_tmp)
            color_tmp = 'blue'

            if activity_tmp == 0:
                color_tmp = 'green'
                total_no_activity+=1
            #elif stdDict[key_tmp] > cutoff:
                # we only analyze when the accuracy STD of the 
                #  trials is greater than 1%
                #if acc_tmp < avgDict[key_tmp]:
            elif acc_tmp < (0.5+cutoff):
                # if the accuracy is below average, then we are
                #  marking this as a bad trial (bad accuracy)
                color_tmp = 'red'
                total_bad_trials+=1
            

            x.append(activity_tmp)
            y.append(acc_tmp)
            c.append(color_tmp)
            total_trials+=1

        plt.clf()
        fig, axs = plt.subplots(1,1)
        scatter = plt.scatter(x,y,color=c,alpha=0.3)
        plt.xlabel('Total Hidden Layer Spikes')
        plt.ylabel('Accuracy')
        plt.ylim([0,1])
        plt.xlim([0,1.75e7])
        axs.set_box_aspect(1)
        plt.tight_layout()
        plt.title(title+' STD cutoff='+str(cutoff)+' ('+str(total_bad_trials)+'bad/'+str(total_trials)+'total)')
        plt.savefig(plotFileBase+'_'+str(cutoff*100)+'_good_bad.png',dpi=600)
        plt.savefig(plotFileBase+'_'+str(cutoff*100)+'_good_bad.svg')
        print("Plot Saved")

        
        

# baseFolder = '/mnt/l/Grad School/Research/Repos/ICONS-data/sparseVaryExtraTemp/'
# dataSet = 'sparseVary_10class'
# logFolder = baseFolder+dataSet+'/logs'
# objFolder = baseFolder+dataSet+'/obj'
# combinedLog = logFolder+'/'+dataSet+'_combined.csv'

# combinedLogFiles = dataSet+'_combined.log'
# read_files = glob.glob(logFolder+'/*.log')
# with open(logFolder+'/'+combinedLogFiles, 'wb') as outfile:
#     for f in read_files:
#         with open(f,'rb') as infile:
#             outfile.write(infile.read())
# simLogger.convertLog(logfile=combinedLogFiles, logdir=logFolder)

# df1 = pd.read_csv(combinedLog)
# df1['color'] = df1.apply(lambda row: getColors(row), axis=1)
# simTags = df1["simTag"].values.tolist()
# fails = []
# writeSparseTestActivityHeader(outputFile=baseFolder+dataSet+'/'+dataSet+'_activity.csv')
# for simTag in simTags:
#     print(simTag)
#     try:
#         # collect activity ---
#         w1_tag = df1[df1['simTag']==simTag]['w1_init'].values[0].split('/')[-1]
#         w2_tag = df1[df1['simTag']==simTag]['w2_init'].values[0].split('/')[-1]
#         param_tag = df1[df1['simTag']==simTag]['surrParams_file'].values[0].split('/')[-1]
#         getSparseTestActvity(w1_tag=w1_tag,w2_tag=w2_tag,param_tag=param_tag,outputFile=baseFolder+dataSet+'/'+dataSet+'_activity.csv',obj_path=objFolder)
#     except:
#         fails.append(simTag)

    # plot activity ---
    # try:
    #     generateActivityHistogram(id,trial)
    # except:
    #     print('NO FILE FOR: '+str(id)+'_t'+str(trial))

# print("FAILS")
# for fail in fails:
#     print(fail)

#weightAccCor(logFile='./data/logs/sparseTest_combined.csv', obj_path='/mnt/l/Grad School/Research/Repos/data_archive_2023_06/sparseTest', plotFileBase='./AAAI/weight')
#combineActivityExcInhAcc(combinedLog='./data/logs/sparseExcHid/combinedLog.csv',activityData='./sparseTest95ActivityClassifiedOutput.csv', plotFileBase='./AAAI/activity_acc_95')
#combineActivityAcc(combinedLog='./data/logs/sparseExcHid/combinedLog.csv',activityData='./sparseTest95ActivityClassifiedOutput.csv', plotFileBase='./AAAI/activity_acc_95')
# combineActivityLoss()

# baseFolder = '/mnt/l/Grad School/Research/Repos/ICONS-data/'
# for s in [50,80,95,99]:  
#     dataSet = 'sparseExcHid'+str(s)
#     logFolder = baseFolder+dataSet+'/logs'
#     objFolder = baseFolder+dataSet+'/obj'
#     combinedLog = logFolder+'/'+dataSet+'_combined.csv'
#     activityData = baseFolder+dataSet+'/'+dataSet+'_activity.csv'
#     plotFileBase = baseFolder+dataSet+'/'+dataSet+'_plot'
#     combineActivityAcc(combinedLog=combinedLog,activityData=activityData, plotFileBase=plotFileBase)
#     #calcAccStats(combinedLog=combinedLog,activityData=activityData, plotFileBase=plotFileBase, title=dataSet)


# calculating activity over epochs
# simtags for lowest activity
#   sparse050_10class_108_t0
#   sparse080_10class_112_t4
#   sparse095_10class_115_t7
#   sparse100_10class_111_t3

# simtags for best performance in the 20-25Hz Range
#   sparse050_10class_147_t7
#   sparse080_10class_140_t0
#   sparse095_10class_147_t7
#   sparse100_10class_147_t7

# simtags for ~10Hz
#   sparse050_10class_124_t0
#   sparse080_10class_126_t2
#   sparse095_10class_124_t0
#   sparse100_10class_130_t6

# uniqueID for weights
#  w1 - simTag_w1_e#_b234, simTag_w1_init
#  w2 - simTag_w2_e#_b234, simTag_w2_init

# ------ 50-50 ratio ------
# simTag = 'sparse050_10class_124_t0'
# baseFolder = '/mnt/l/Grad School/Research/Repos/ICONS-data/'
# dataSet = 'sparse050_10class'
# logFolder = baseFolder+dataSet+'/logs'
# objFolder = baseFolder+dataSet+'/obj'
# combinedLog = logFolder+'/'+dataSet+'_combined.csv'
# combinedActvity = baseFolder+'epoch_10HzBase_activity.csv'

# writeSparseTestActivityHeader(outputFile=combinedActvity)
# w1_tag = simTag+'_w1_init'
# w2_tag = simTag+'_w2_init'
# param_tag = simTag+'_surrParams'
# getSparseTestActvity(w1_tag=w1_tag,w2_tag=w2_tag,param_tag=param_tag,outputFile=combinedActvity,obj_path=objFolder)

# for e in range(30):
#     w1_tag = simTag+'_w1_e'+str(e)+'_b234'
#     w2_tag = simTag+'_w2_e'+str(e)+'_b234'
#     getSparseTestActvity(w1_tag=w1_tag,w2_tag=w2_tag,idTag=simTag+'_e'+str(e), param_tag=param_tag,outputFile=combinedActvity,obj_path=objFolder)


# # ------ 80-20 ratio ------
# simTag = 'sparse080_10class_126_t2'
# baseFolder = '/mnt/l/Grad School/Research/Repos/ICONS-data/'
# dataSet = 'sparse080_10class'
# logFolder = baseFolder+dataSet+'/logs'
# objFolder = baseFolder+dataSet+'/obj'
# combinedLog = logFolder+'/'+dataSet+'_combined.csv'

# w1_tag = simTag+'_w1_init'
# w2_tag = simTag+'_w2_init'
# param_tag = simTag+'_surrParams'
# getSparseTestActvity(w1_tag=w1_tag,w2_tag=w2_tag,param_tag=param_tag,outputFile=combinedActvity,obj_path=objFolder)

# for e in range(30):
#     w1_tag = simTag+'_w1_e'+str(e)+'_b234'
#     w2_tag = simTag+'_w2_e'+str(e)+'_b234'
#     getSparseTestActvity(w1_tag=w1_tag,w2_tag=w2_tag,idTag=simTag+'_e'+str(e), param_tag=param_tag,outputFile=combinedActvity,obj_path=objFolder)


# # ------ 95-5 ratio ------
# simTag = 'sparse095_10class_124_t0'
# baseFolder = '/mnt/l/Grad School/Research/Repos/ICONS-data/'
# dataSet = 'sparse095_10class'
# logFolder = baseFolder+dataSet+'/logs'
# objFolder = baseFolder+dataSet+'/obj'
# combinedLog = logFolder+'/'+dataSet+'_combined.csv'

# w1_tag = simTag+'_w1_init'
# w2_tag = simTag+'_w2_init'
# param_tag = simTag+'_surrParams'
# getSparseTestActvity(w1_tag=w1_tag,w2_tag=w2_tag,param_tag=param_tag,outputFile=combinedActvity,obj_path=objFolder)

# for e in range(30):
#     w1_tag = simTag+'_w1_e'+str(e)+'_b234'
#     w2_tag = simTag+'_w2_e'+str(e)+'_b234'
#     getSparseTestActvity(w1_tag=w1_tag,w2_tag=w2_tag,idTag=simTag+'_e'+str(e), param_tag=param_tag,outputFile=combinedActvity,obj_path=objFolder)

# ------ 100-0 ratio ------
# simTag = 'sparse100_10class_130_t6'
# baseFolder = '/mnt/l/Grad School/Research/Repos/ICONS-data/'
# dataSet = 'sparse100_10class'
# logFolder = baseFolder+dataSet+'/logs'
# objFolder = baseFolder+dataSet+'/obj'
# combinedLog = logFolder+'/'+dataSet+'_combined.csv'
# w1_tag = simTag+'_w1_init'
# w2_tag = simTag+'_w2_init'
# param_tag = simTag+'_surrParams'
# # getSparseTestActvity(w1_tag=w1_tag,w2_tag=w2_tag,param_tag=param_tag,outputFile=combinedActvity,obj_path=objFolder)

# for e in range(3):
#     w1_tag = simTag+'_w1_e'+str(e)+'_b234'
#     w2_tag = simTag+'_w2_e'+str(e)+'_b234'
#     getSparseTestActvity(w1_tag=w1_tag,w2_tag=w2_tag,idTag=simTag+'_e'+str(e), param_tag=param_tag,outputFile=combinedActvity,obj_path=objFolder)
