from weightAnalysis import getWeightFromUniqueId

baseSimTag = 'testNoise4'
objFolder = './data/testNoise4/obj'

w = []
pre = 400
post = 34

while(pre!=-1 and post!=-1):
    w1 = getWeightFromUniqueId(baseSimTag+'_w1_init',objFolder=objFolder)
    w.append(w1[pre][post])
    for e in range(20):
        w1 = getWeightFromUniqueId(baseSimTag+'_w1_e'+str(e),objFolder=objFolder)
        w.append(w1[pre][post])
        
    
    for e in range(-1, 20):
        print('EPOCH: '+str(e)+'\tW: '+str(w[e+1]))

    pre = int(input('ENTER PRE SYNAPTIC INDEX:'))
    post = int(input('ENTER POST SYNAPTIC INDEX:'))
