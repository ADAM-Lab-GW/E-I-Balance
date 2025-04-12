import surrogateTraining
from surrogateParams import surrParams

parameter_name = "CA3 Pyramidal Subtype 3"
k=0.7996081098116854
a=0.005872238026714838
b=-42.5524776883928
d=588.0
C=41.0
vRest=-55.361334402524086
vThresh=-20.08942170607665
vMin=-38.8680990294091
vPeak=35.8614648558726

params = surrParams()
params.simTag = 'testIZ'
nIdMin = 0
nIdMax = 80
isExc = True
params.setIZParams(C,k,vMin,vRest,vThresh,vPeak,a,b,d,isExc,nIdMin,nIdMax)
nIdMin = 80
nIdMax = 100
isExc = False
params.setIZParams(C,k,vMin,vRest,vThresh,vPeak,a,b,d,isExc,nIdMin,nIdMax)
params.debug = True

acc,loss = surrogateTraining.main(params)
print(acc)
print(loss)