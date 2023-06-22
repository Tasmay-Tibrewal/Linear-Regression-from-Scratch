import csv
import math
import matplotlib.pyplot as plt
import numpy as np





def read():

    global l,header

    with open ('Stock Regression\MSFT_daily_dataset_1year.csv','r',newline='') as csvfile:

        csvreader = csv.reader(csvfile)

        header = csvreader.__next__()

        for i in csvreader:
            l.append(i)





def read2():

    global l2

    with open ('Stock Regression\MSFT _daily_dataset_test.csv','r',newline='') as csvfile:

        csvreader = csv.reader(csvfile)

        header = csvreader.__next__()

        for i in csvreader:
            l2.append(i)





def points_change():

    global l, ymin, ymax, xpoints, ypoints, ypoints2

    for i in l:

        if(float(i[-2]) < ymin):
            ymin = float(i[-2])

        elif(float(i[-2]) > ymax):
            ymax = float(i[-2])

        ypoints.append(float(i[-2]))
        xpoints.append(float(i[-1]))

        i[-2] = str(-1.03*float(i[-2])+float(i[-3]))

        ypoints2.append(float(i[-2]))





def sum_avg_std():

    global sum, avg, stdev,n,l

    for i in l:
        for j in range(6):
            sum[j] += float(i[j+1])
        n += 1

    for i in range(6):
        avg[i] = sum[i]/n

    for i in  l:
        for j in range(6):
            stdev[j] += (float(i[j+1]) - avg[j])**2/n

    for j in range(6):
        stdev[j] = math.sqrt(stdev[j])





def cor_cov():

    global cov, cor, h1, h2, l, header, avg, n, stdev

    cov.append(h1)
    cor.append(h2)

    count  = 0
    for i in l:
        count += 1
        for j in range(6):
            if (count == 1):
                cov.append([0]*7)
                cor.append([0]*7)
                cov[j+1][0] = header[j+1]
            for k in range(6): 
                cov[j+1][k+1] += (float(i[j+1]) - avg [j]) * (float(i[k+1]) - avg [k]) / n
            
    for j in range(6):
        cor[j+1][0] = cov[j+1][0]
        for k in range(6):
            cor[j+1][k+1] += cov[j+1][k+1] / (stdev[j] * stdev[k])





#code for linear regression not required as least square method is sufficient
def cost_fn(o1,o0):

    global l,n

    cost = 0

    for i in range(n):
        cost += ( 1 / (2*n)) * ((float(l[i][-2]) - o1*float(l[i][-1]) -o0) ** 2)
    
    return cost





def diff_o1(o1,o0):
    
    global l,n

    diff = 0

    for i in range(n):
        diff += (1/n) * (float(l[i][-2]) - o1*float(l[i][-1]) - o0) * -1 * float(l[i][-1])

    return diff





def diff_o0(o1,o0):
    
    global l,n

    diff = 0

    for i in range(n):
        diff += (1/n) * (float(l[i][-2]) - o1*float(l[i][-1]) - o0) * -1

    return diff





def lsf():

    global avg, cor, cov, stdev, o1, o0, n, l

    price = avg[-2]

    x = avg[-1]

    #least square fitting method
    o1 = cor[-1][-2] * stdev[-2] / stdev[-1]
    o0 = (price - o1 * x)

    print()
    print("o1 : ",o1)
    print("o0 : ",o0)
    print()

    cost = cost_fn(o1,o0)
    diff1 = diff_o1(o1,o0)
    diff0 = diff_o0(o1,o0)
    costcopy = cost
    a = 3.9*math.pow(10,-11)

    print("COST : ",cost)
    #print(diff1)
    #print(diff0)
    print("Alpha : ",a)
    print()
    

    o1 = o1 - diff1 * a
    o0 = o0 - diff0 * a

    print()
    print("o1 [new] : ",o1)
    print("o0 [new] : ",o0)
    print()

    while (cost - costcopy > 0.001 or cost - costcopy < -0.001):

        costcopy = cost

        o1 = o1 - diff1 * a
        o0 = o0 - diff0 * a

        cost = cost_fn(o1,o0)
        diff1 = diff_o1(o1,o0)
        diff0 = diff_o0(o1,o0)

        print("Cost : ",cost)
        #print(diff1)
        #print(diff0)
        print()

    print()
    print("Final Cost : ",cost)
    print("o1 [final] : ",o1)
    print("o0 [final] : ",o0)
    print()
    print()







def regress():

    global avg, cor, cov, stdev, o1, o0, n, l

    costs = [-1,0,0,0,0,0]
    costlast = -1

    for i in range(-128,64):

        if i > 32:
            i2 = i - 32
            o1 = -math.pow(10,(i2/8))
        
        elif i < -64:
            i2 = i + 64
            o1 = -math.pow(10,(i2/8))

        else:
            o1 = math.pow(10,(i/8))

        for k in range(-128,128):

            if k > 64:
                k2 = k - 64
                o0 = -math.pow(10,(k2/16))
            
            elif k < -64:
                k2 = k + 64
                o0 = -math.pow(10,(k2/16))

            else:
                o0 = math.pow(10,(k/16))

            o1copy = o1
            o0copy = o0

            cost = cost_fn(o1,o0)
            diff1 = diff_o1(o1,o0)
            diff0 = diff_o0(o1,o0)
            costin = cost
            costcopy = cost
            diff1in = diff1
            diff0in = diff0
            costmin = cost

            if cost > costs[0] and abs(diff1in) <= costs[4] and abs(diff0in) <= costs[5]:
                continue

            if cost - costlast > 0.01 or cost - costlast < -0.01:
                costlast = cost

            elif costlast == -1 :
                costlast = cost

            else:
                continue

            if costs[0] == -1:
                costs[0] = cost
                costs[1] = o1
                costs[2] = o0
                costs[3] = costin
                costs[4] = abs(diff1in)
                costs[5] = abs(diff0in)
                
            elif costs[0] > cost :
                costs[0] = cost
                costs[1] = o1
                costs[2] = o0
                costs[3] = costin
                costs[4] = abs(diff1in)
                costs[5] = abs(diff0in)

            start = -16.25
            end = 8.25

            while((end - start) > 0.25):

                mid = (start + end) / 2

                a = math.pow(10,(mid/4))
                j = 0
                count = 0
                

                o1 = o1 - diff1 * a
                o0 = o0 - diff0 * a

                while (cost - costcopy > 0.01 or cost - costcopy < -0.01):

                    j += 1

                    if (j>20):
                        break

                    if (costcopy < cost):
                        cost = costcopy
                        count = 1
                        break

                    costcopy = cost

                    o1 = o1 - diff1 * a
                    o0 = o0 - diff0 * a

                    cost = cost_fn(o1,o0)
                    diff1 = diff_o1(o1,o0)
                    diff0 = diff_o0(o1,o0)

                if count == 1:
                    end = mid

                else:
                    start = mid

                if(cost<costmin):
                    costmin = cost

                if costs[0] > cost :
                    costs[0] = cost
                    costs[1] = o1
                    costs[2] = o0
                    costs[3] = costin
                    costs[4] = abs(diff1in)
                    costs[5] = abs(diff0in)
            
                o1 = o1copy
                o0 = o0copy
    
    o1 = costs[1]
    o0 = costs[2]
    cost = costs[0]

    print()
    print("Final Cost : ",cost)
    print("o1 [final] : ",o1)
    print("o0 [final] : ",o0)
    print()
    print()


def regress_new():

    global avg, cor, cov, stdev, o1, o0, n, l

    price = avg[-2]

    x = avg[-1]

    #least square fitting method
    o1 = cor[-1][-2] * stdev[-2] / stdev[-1]
    o0 = (price - o1 * x)





def fn_basic(o1,o0):
    y_calc = []
    for i in l:
        y_calc.append(o1 * float(i[-1]) + o0)
    return y_calc





def fn_final(o1,o0):
    y_calc = []
    for i in l:
        y_calc.append((float(i[-3]) - o1 * float(i[-1]) - o0)/1.03)
    return y_calc





def mse(o1,o0):

    global l,n

    mse = 0

    for i in range(n):
        mse += ((float(l[i][-2]) - o1*float(l[i][-1]) -o0) ** 2)/n
    
    return mse





def rmse_train(o1,o0):

    global l,n

    rmse = 0

    for i in range(n):
        rmse += ((float(l[i][-2]) - o1*float(l[i][-1]) -o0) ** 2)/n

    rmse = math.sqrt(rmse)
    
    return rmse





def rmse_test(o1,o0):

    global l2,n

    rmse = 0

    for i in range(len(l2)):
        rmse += ((float(l2[i][-2]) - o1*float(l2[i][-1]) -o0) ** 2)/n

    rmse = math.sqrt(rmse)
    
    return rmse





def plot():
    xpoints_arr = np.array(xpoints)
    ypoints_arr = np.array(ypoints)

    xpoints_arr_ck = np.array([xpoints[14], xpoints[16], xpoints[156], xpoints[157], xpoints[205], xpoints[229], xpoints[238], xpoints[241]])
    ypoints2_arr_ck = np.array([ypoints2[14], ypoints2[16], ypoints2[156], ypoints2[157], ypoints2[205], ypoints2[229], ypoints2[238], ypoints2[241]])

    ypoints_calc = fn_final(o1,o0)
    ypoints_arr_calc = np.array(ypoints_calc)

    ypoints2_calc = fn_basic(o1,o0)
    ypoints2_arr_calc = np.array(ypoints2_calc)

    ypoints2_arr = np.array(ypoints2)

    cook_dist_arr = np.array(cook_dist)

    #for i in range(n):
    #    print(ypoints_calc[i],"    ",ypoints[i]," Error : ",abs(ypoints_calc[i]-ypoints[i]))

    plt.plot(xpoints_arr, ypoints_arr,'o')
    plt.plot(xpoints_arr, ypoints_arr_calc)

    plt.figure()

    plt.plot(xpoints_arr, ypoints2_arr,'o')
    plt.plot(xpoints_arr, ypoints2_arr_calc)
    plt.plot(xpoints_arr_ck, ypoints2_arr_ck,'o',color = 'red')

    plt.figure()

    for i in range(n):
        plt.plot(np.array([xpoints[i],xpoints[i]]), np.array([0,cook_dist[i]]),color = 'blue')

    plt.plot(xpoints_arr,cook_dist_arr,'.')

    plt.axhline(y=3 * mean_cook_dist, linestyle = 'dashed')

    plt.show()






def new_model():
    
    global header, l, sum, avg, stdev, cov, cor, xpoints, ypoints, ypoints2, n, o1, o0, h1, h2

    header2 = header.copy() 
    l2 = l.copy()
    sum2 = sum.copy()
    avg2 = avg.copy()
    stdev2 = stdev.copy()
    cov2 = cov.copy()
    cor2 = cor.copy()
    n2 = n
    o12 = o1
    o02 = o0
    h12 = h1.copy()
    h22 = h2.copy()
    
    sum = [0] * 6
    avg = [0] * 6
    stdev = [0] * 6
    cov = []
    cor = []
    n = 0
    o1 = 0
    o0 = 0
    h1 = ['Covariance Matrix']
    h2 = ['Correlation Matrix']

    h1.extend(header[1:])
    h2.extend(header[1:])

    sum_avg_std()

    cor_cov()

    regress_new()

    x = o1
    c = o0

    header = header2.copy() 
    l = l2.copy()
    sum = sum2.copy()
    avg = avg2.copy()
    stdev = stdev2.copy()
    cov = cov2.copy()
    cor = cor2.copy()
    n = n2
    o1 = o12
    o0 = o02
    h1 = h12.copy()
    h2 = h22.copy()

    return x,c





def cook_distance():

    global cook_dist, l, n, o0, o1

    s2 = mse(o1,o0)

    l3 = l.copy()

    for i in range(len(l3)):

        distance = 0
        
        l.pop(i)

        o12 , o02 = new_model()

        l = l3.copy()

        for j in range(len(l3)):

            val_rem = o12*float(l3[j][-1]) + o02

            distance += ((o1*float(l3[j][-1]) + o0 - val_rem)**2)/s2
        
        cook_dist.append(distance)





def initiate():
    global header, l, sum, avg, stdev, cov, cor, xpoints, ypoints, ypoints2, n, o1, o0, l2, h1, h2, cook_dist, mean_cook_dist, outliers, ymin, ymax

    header = [] 
    l = []
    sum = [0] * 6
    avg = [0] * 6
    stdev = [0] * 6
    cov = []
    cor = []
    xpoints = []
    ypoints = []
    ypoints2 = []
    n = 0
    o1 = 0
    o0 = 0
    l2 = []
    h1 = ['Covariance Matrix']
    h2 = ['Correlation Matrix']
    cook_dist = []
    mean_cook_dist = 0
    outliers = []

    read()

    ymin = float(l[0][-2])
    ymax = float(l[0][-2])

    h1.extend(header[1:])
    h2.extend(header[1:])

    points_change()

    sum_avg_std()

    print("SUM : ",sum)
    print("AVG : ",avg)
    print("STD. DEV : ",stdev)
    print()

    cor_cov()

    print("COV MATRIX : ")
    for i in cov:
        print(i)

    print()

    print("COR MATRIX : ")
    for i in cor:
        print(i)

    print()
    print("\nPlease wait for 15-20 seconds\n")

    regress()

    error = rmse_train(o1,o0)

    print()
    print("RMSE ERROR on TRAIN : ",error)
    print()

    read2()

    for i in l2:

        i[-2] = str(-1.03*float(i[-2])+float(i[-3]))


    error = rmse_test(o1,o0)

    print()
    print("RMSE ERROR on TEST : ",error)
    print()
    print()

    cook_distance()

    for i in cook_dist:
        mean_cook_dist += i

    mean_cook_dist /= n

    #print("COOKS DISTANCES/MEAN :")
    #for i in range(len(cook_dist)):
    #    print(i," : ",cook_dist[i]/mean_cook_dist)

    for i in range(n):
        if(cook_dist[i] > 3 * mean_cook_dist):
            outliers.append(i)

    print("OUTLIERS : ",outliers)
    print("Number of Outliers : ",len(outliers))
    print()

    print()
    print()
    print("NEW MODEL : ")
    
    plot()
    
    less = 0

    for i in outliers:
        l.pop(i-less)
        xpoints.pop(i-less)
        ypoints.pop(i-less)
        ypoints2.pop(i-less)
        cook_dist.pop(i-less)
        less+=1

    n -= len(outliers)

    o11 = o1
    o01 = o0

    o1,o0 = new_model()

    print("Old o1 : ",o11,"Old o0 : ",o01)
    print("New o1 : ",o1,"New o0 : ",o0)
    print()
    print()

    '''for i in cov:
        print(i)

    print()

    for i in cor:
        print(i)'''

    error = rmse_train(o1,o0)

    print()
    print("NON-OUTLIERS RMSE ERROR on TRAIN : ",error)
    print()

    l2 = []

    read2()

    for i in l2:

        i[-2] = str(-1.03*float(i[-2])+float(i[-3]))


    error = rmse_test(o1,o0)

    print()
    print("NON-OUTLIERS RMSE ERROR on TEST : ",error)
    print()
    print()

    plot()




initiate()