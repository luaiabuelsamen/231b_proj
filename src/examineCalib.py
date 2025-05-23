import numpy as np
import scipy as sp
import scipy.linalg as linalg
import scipy.stats as stats
import matplotlib.pyplot as plt

from scipy.stats import norm

datafile = 'data/run_000.csv'

experimentalData = np.genfromtxt (datafile, delimiter=',')


t = experimentalData[:,0]
gamma = experimentalData[:,1]  #steering able.
omega = experimentalData[:,2]  #pedaling speed
measx = experimentalData[:,3]  #gps X
measy = experimentalData[:,4]  #gps Y


final_x = experimentalData[-1,5]
final_y = experimentalData[-1,6]


print(gamma)
#print(omega)
#print


if datafile == 'data/run_000.csv':
	assert np.all(gamma[:]==0), 'Non-zero steering angles detected in the calibration data!'
	assert np.all(omega[:]==0), 'Non-zero pedal speeds detected in the calibration data!'



print("Known Bike position:",final_x,final_y)


print("Number of GPS Measurements: ",np.count_nonzero(~np.isnan(measx)))
print("GPS Means\t",np.nanmean(measx),np.nanmean(measy))
print("GPS STD\t\t",np.nanstd(measx),np.nanstd(measy))
print("GPS VAR\t\t",np.nanvar(measx),np.nanvar(measy))
print("GPS BIAS\t",final_x-np.nanmean(measx),final_y-np.nanmean(measy))
print("GPS Normal Test",stats.normaltest(np.nanmean(measx)-measx,nan_policy='omit'),stats.normaltest(np.nanmean(measy)-measy,nan_policy='omit'))
print("--------------    GPS Biased?  --------------")
print("GPS ERROR\t\t",np.nanmean(measx)-final_x,np.nanmean(measy)-final_y)
print("GPS Standard Error of the Mean",np.nanstd(measx-final_x)/np.sqrt(np.count_nonzero(~np.isnan(measx))),np.nanstd(measy-final_y)/np.sqrt(np.count_nonzero(~np.isnan(measy))))
print("N Sigma:\t\t",(np.nanmean(measx)-final_x)/(np.nanstd(measx)/np.sqrt(np.count_nonzero(~np.isnan(measx)))),(np.nanmean(measy)-final_y)/(np.nanstd(measy)/np.sqrt(np.count_nonzero(~np.isnan(measy)))))


print('------   Analysis to See if Errors are Indpendant   ----------')
non_nan_locs = np.where(~np.isnan(measx))

errs_x = measx[non_nan_locs]-final_x
errs_y = measy[non_nan_locs]-final_y
print("Covariance Matrix: ")
print(np.cov(np.vstack((errs_x,errs_y))))

print("----------")
print("Average Period:",np.mean(np.diff(t[non_nan_locs])))

plt.figure()
plt.subplot(3,1,1)
plt.hist(measx,bins=50,density=True)
plt.axvline(x=np.nanmean(measx),c='b',label='E[GPS_X]')
plt.axvline(x=final_x,c='g',label='True GPS X')
plt.axvline(x=np.nanmean(measx)-np.nanstd(measx),c='r',ls='-.')
plt.axvline(x=np.nanmean(measx)+np.nanstd(measx),label=r'$\pm$ 1 standard deviation',c='r',ls='-.')
plt.xlabel("X Error [m]")
plt.ylabel("PDF Density")
#overlay a gaussian plot with the same mean and norm.
x_pts = np.linspace(np.nanmin(measx),np.nanmax(measx))
plt.plot(x_pts,norm.pdf(x_pts,loc=np.nanmean(measx),scale=np.nanstd(measx)),c='c',label='Normal Distribution from data')

plt.legend()

plt.subplot(3,1,2)
plt.hist(measy,bins=50,density=True)
plt.axvline(x=np.nanmean(measy),c='b',label='E[GPS_Y]')
plt.axvline(x=final_y,c='g',label='True GPS y')
plt.axvline(x=np.nanmean(measy)-np.nanstd(measy),c='r',ls='-.')
plt.axvline(x=np.nanmean(measy)+np.nanstd(measy),label=r'$\pm$ 1 standard deviation',c='r',ls='-.')
#overlay a gaussian plot with the same mean and norm.
y_pts = np.linspace(np.nanmin(measy),np.nanmax(measy))
plt.plot(y_pts,norm.pdf(y_pts,loc=np.nanmean(measy),scale=np.nanstd(measy)),c='c',label='Normal Distribution from data')
plt.xlabel("Y Error [m]")
plt.ylabel("PDF Density")




plt.legend()
plt.subplot(3,1,3)
plt.hist(np.diff(t[non_nan_locs]),bins=10,density=True)
plt.axvline(x=0.5,c='b',label='Nominal GPS Period')
plt.axvline(x=np.mean(np.diff(t[non_nan_locs])),c='g',label='Mean GPS Period')

plt.legend()
plt.xlabel("Data Period [sec]")
plt.ylabel("PDF Density")



plt.show()