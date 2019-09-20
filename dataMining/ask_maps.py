import pandas as  pd
import time
from ast import literal_eval
from gmplot import gmplot
import numpy as np
from haversine import haversine #project PyPi haversine
from fastdtw import fastdtw
from sklearn.model_selection import KFold
import operator




#################################################################
#Q1
#################################################################
trainSet = pd.read_csv(
    './train_set.csv',
    converters={"Trajectory":literal_eval},
    index_col='tripId'
)
trajS_train=list(trainSet['Trajectory'])
x=5 #pente protes fores
for f in range(x):
    y=len(trajS_train[f]) #len of each line
    pathlon = list() #kathe fora dimourgoume th lista me
    pathlat = list() #ta zeugaria lat lon mias diadromis
    for i in range(y):
        pathlon.append(trajS_train[f][i][1])
        pathlat.append(trajS_train[f][i][2])

    gmap = gmplot.GoogleMapPlotter(pathlat[0],pathlon[0],18)
    gmap.plot(pathlat,pathlon,'cornflowerblue',edge_width=10)
    m = "map%d.html" % (f) #sprintf
    gmap.draw(m)

#################################################################
#Q2
#################################################################

testSet1 = pd.read_csv('./test_set_a1.csv',sep='\t',converters={"Trajectory":literal_eval})
trajS_test_a1 = list(testSet1['Trajectory'])
len_train = len(trajS_train)
len_test = len(trajS_test_a1)
JPid = list(trainSet['journeyPatternId']) #lisTA me ta jp id
counterQ2=0 #metritis html
dt_list = list() #lista mw tous xronous

for d in range(len_test):
	#ftiaxnoume to test_array_np
	start_time = time.time()
	len_test_line = len(trajS_test_a1[d])
	test_array = [[0 for ai in range(2)]for aj in range(len_test_line)]
	for ii in range(len_test_line):
		test_array[ii][0] = trajS_test_a1[d][ii][2] #lat
		test_array[ii][1] =	trajS_test_a1[d][ii][1] #lon
	test_array_np = np.array(test_array)
	#ftiaxounme to train_array_np
	my_min_list = list()
	for ff in range(len_train): #len_train
		yy=len(trajS_train[ff]) #len of each line
		train_array = [[0 for i in range(2)]for j in range(yy)]
		for jj in range(yy):
			train_array[jj][0] =trajS_train[ff][jj][2] #lat
			train_array[jj][1] =trajS_train[ff][jj][1] #lon
		train_array_np = np.array(train_array)
		distance,path = fastdtw(test_array_np,train_array_np,dist=haversine)
		#print(distance)
		my_min_list.append((distance,ff))
	#sort the my_min_list
	my_min_list.sort(key=lambda x:x[0])
	print "Test Trajectory path:",d
	print "Neighbour:1","DTW,traincsv_pos",my_min_list[0],"JP ID:",JPid[my_min_list[0][1]]
	print "Neighbour:2","DTW,traincsv_pos",my_min_list[1],"JP ID:",JPid[my_min_list[1][1]]
	print "Neighbour:3","DTW,traincsv_pos",my_min_list[2],"JP ID:",JPid[my_min_list[2][1]]
	print "Neighbour:4","DTW,traincsv_pos",my_min_list[3],"JP ID:",JPid[my_min_list[3][1]]
	print "Neighbour:5","DTW,traincsv_pos",my_min_list[4],"JP ID:",JPid[my_min_list[4][1]]
	pathlonq2test = list() #kathe fora dimourgoume th lista me
	pathlatq2test = list() #ta zeugaria lat lon mias diadromis
	for r in range(len_test_line):
		pathlonq2test.append(trajS_test_a1[d][r][1])
		pathlatq2test.append(trajS_test_a1[d][r][2])
	gmap = gmplot.GoogleMapPlotter(pathlatq2test[0],pathlonq2test[0],18)
	gmap.plot(pathlatq2test,pathlonq2test,'cornflowerblue',edge_width=10)
	m_test = "Q2mapTest%d.html" % (d) #sprintf
	gmap.draw(m_test)
	#ftiaxume html gia test diadromi kai 5 geitones
	xx=5 #pente protes fores
	for q2 in range(xx):
		y=len(trajS_train[my_min_list[q2][1]]) #len of each line
		pathlonq2 = list() #kathe fora dimourgoume th lista me
		pathlatq2 = list() #ta zeugaria lat lon mias diadromis
		for i in range(y):
			pathlonq2.append(trajS_train[my_min_list[q2][1]][i][1])
			pathlatq2.append(trajS_train[my_min_list[q2][1]][i][2])

		gmap = gmplot.GoogleMapPlotter(pathlatq2[0],pathlonq2[0],18)
		gmap.plot(pathlatq2,pathlonq2,'cornflowerblue',edge_width=10)
		mQ2 = "Q2map%d.html" % (counterQ2) #sprintf
        	counterQ2+=1
        	gmap.draw(mQ2)
        	#time.sleep(3)
	end_time = time.time()
	dt_list.append(end_time-start_time)
	print "dt:",end_time-start_time

testSet2 = pd.read_csv('./test_set_a2.csv',sep='\t',converters={"Trajectory":literal_eval})
trajS_test_a2 = list(testSet2['Trajectory'])
len_test_a2 = len(trajS_test_a2)

def my_LCS(list_x,list_y):
	m = len(list_x)
	n = len(list_y)
	# An (m+1) times (n+1) matrix
	C = [[0] * (n + 1) for _ in range(m + 1)]
	for i in range(1, m+1):
		for j in range(1, n+1):	
			dist_haversine = haversine(list_x[i-1],list_y[j-1])
			if dist_haversine < 0.2 : 
				C[i][j] = C[i-1][j-1] + 1
			else:
				C[i][j] = max(C[i][j-1], C[i-1][j])
	return C

def my_backTrack(C, X, Y, i, j):
	lcs_list = list()
	while i >0 and j >0:
		dist_haversine = haversine(X[i-1],Y[j-1])
		if dist_haversine < 0.2:
			lcs_list.append(X[i-1])
			i-=1
			j-=1
		elif C[i-1][j] > C[i][j-1]:
			i-=1
		else:
			j-=1	
	return lcs_list

for c in range(len_test_a2): # len test_a2
	start_time=time.time()
	len_test_line_a2 = len(trajS_test_a2[c])
	list_x = list()
	my_max_list = list()
	for u in range(len_test_line_a2):
		list_x.append((trajS_test_a2[c][u][2],trajS_test_a2[c][u][1]))
	#exoume etoimi th proti lat,lon lista
	for cc in range(len_train): #len train
		list_y = list()
		ll = len(trajS_train[cc])
		for uu in range(ll):
			list_y.append((trajS_train[cc][uu][2],trajS_train[cc][uu][1]))
			#exoume etoimi kai th deuteri lista 
		matrix_lcs = my_LCS(list_x,list_y)
		matching_path = my_backTrack(matrix_lcs,list_x,list_y,len(list_x),len(list_y))
		
		correct_mp = matching_path[::-1] # reverse list trick
		if len(correct_mp)!=0:
			#print len(correct_mp)
			my_max_list.append((len(correct_mp),cc,correct_mp))

	#sort max list me vasi to matching points
	my_max_list.sort(key=lambda x:x[0])
	max_list = list()
	max_list.append(my_max_list[len(my_max_list)-5])
	max_list.append(my_max_list[len(my_max_list)-4])
	max_list.append(my_max_list[len(my_max_list)-3])
	max_list.append(my_max_list[len(my_max_list)-2])
	max_list.append(my_max_list[len(my_max_list)-1])
	print "Test Trajectory path:",c
	print "Neighbour:1","MP",max_list[0][0],"JP ID:",JPid[max_list[0][1]]
	print "Neighbour:2","MP",max_list[1][0],"JP ID:",JPid[max_list[1][1]]
	print "Neighbour:3","MP",max_list[2][0],"JP ID:",JPid[max_list[2][1]]
	print "Neighbour:4","MP",max_list[3][0],"JP ID:",JPid[max_list[3][1]]
	print "Neighbour:5","MP",max_list[4][0],"JP ID:",JPid[max_list[4][1]]

	pathlonq2a2test = list() #kathe fora dimourgoume th lista me
	pathlatq2a2test = list() #ta zeugaria lat lon mias diadromis
	for rr in range(len_test_line_a2):
		pathlonq2a2test.append(trajS_test_a2[c][rr][1])
		pathlatq2a2test.append(trajS_test_a2[c][rr][2])
	gmap = gmplot.GoogleMapPlotter(pathlatq2a2test[0],pathlonq2a2test[0],18)
	gmap.plot(pathlatq2a2test,pathlonq2a2test,'cornflowerblue',edge_width=10)
	m_test_a2 = "Q2A2mapTest%d.html" % (c) #sprintf
	gmap.draw(m_test_a2)
	xxx=5 #pente protes fores
	for q2a2 in range(xxx):
		y2=len(trajS_train[max_list[q2a2][1]]) #len of each line
		pathlonq2a2 = list() #kathe fora dimourgoume th lista me
		pathlatq2a2 = list() #ta zeugaria lat lon mias diadromis
		for i2 in range(y2):
			pathlonq2a2.append(trajS_train[max_list[q2a2][1]][i2][1])
			pathlatq2a2.append(trajS_train[max_list[q2a2][1]][i2][2])

		cp_pathlon = list()
		cp_pathlat = list()
		for i3 in range(max_list[q2a2][0]):
			cp_pathlon.append(max_list[q2a2][2][i3][1])
			cp_pathlat.append(max_list[q2a2][2][i3][0])

		gmap = gmplot.GoogleMapPlotter(pathlatq2a2[0],pathlonq2a2[0],18)
		gmap.plot(pathlatq2a2,pathlonq2a2,'cornflowerblue',edge_width=10)
		gmap.plot(cp_pathlat,cp_pathlon,'red',edge_width=10)
		mQ2 = "Q2map%d.html" % (counterQ2) #sprintf
        	counterQ2+=1
        	gmap.draw(mQ2)
        	#time.sleep(3)
	end_time = time.time()
	dt_list.append(end_time-start_time)
	print "dt:",end_time-start_time

#################################################################
#Q3
#################################################################

#exoume to trainSet kai to testSet2
#kai xrisimopoioume to haversine dist 
#xoume to trajS_train me trajectory tou train 
#kai to trajS_test_a2 me trajectory tou test 	
def getResult(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(),key=operator.itemgetter(1),reverse=True)
	return sortedVotes[0][0]
	
kf = KFold(n_splits = 10)

print("\nMaking the csv")
result_list = list()
index_list = list()

for fd in range(len(trajS_test_a2)):
	k=5 # knn
	neighbors = list()
	distances = list()
	y_len_s = len(trajS_test_a2[fd])
	te_array = [[0 for ai in range(2)]for aj in range(y_len_s)]
	#print fd,y_len_s
	for gg in range(y_len_s):
		te_array[gg][0] = trajS_test_a2[fd][gg][2] #lat
		te_array[gg][1] = trajS_test_a2[fd][gg][1] #lon
	te_array_np = np.array(te_array)
		
	for g in range(len(trajS_train)):
		y = len(trajS_train[g])
		tr_array = [[0 for ai in range(2)]for aj in range(y)]
		for gg in range(y):
			tr_array[gg][0] = trajS_train[g][gg][2]#lat
			tr_array[gg][1] = trajS_train[g][gg][1] #lon
		tr_array_np = np.array(tr_array)
		distance,path = fastdtw(te_array_np,tr_array_np,dist=haversine)
		distances.append((distance,JPid[g]))
	distances.sort(key=lambda x:x[0])	
	for x in range(k):
		neighbors.append((distances[x][0],distances[x][1])) #to duetero einai to JPid
	
	result = getResult(neighbors) #result einai to JPid pou anikei to test trajectory
	print result
	result_list.append(result)
	index_list.append(fd)
	index_list[fd] += 1


result_array = np.transpose([index_list,result_list])
result_np = np.array(result_array)
df_result = pd.DataFrame(result_np)
df_result.columns = ['Test_Trip_ID','Predicted_JourneyPatternID']
df_result.to_csv("testSet_JourneyPatternIDs.csv",index = False ,sep='\t')







