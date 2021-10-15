import agent1,agent2,agent3,agent4,astar,time
import numpy as np
import matplotlib.pylab as plt

def creategrid(dim,p):   
    grid = [[0 for i in range(dim)] for j in range(dim)]
    for i in range(dim):
        for j in range(dim):
            actual_prob = np.random.random_sample()  ## generating a random number
            if actual_prob > p:                      ## if the generated random number > p, assign it 1 (meaning it is
                grid[i][j] = 1                       ## traversable.
            else:
                grid[i][j] = 0

    grid[0][0] = 1                                   ## start node and end node is always traversable.
    grid[dim - 1][dim - 1] = 1
    return grid


timetaken_agent1=[]
timetaken_agent2=[]
timetaken_agent3=[]
timetaken_agent4=[]
avg_timetaken_agent1=[]
avg_timetaken_agent2=[]
avg_timetaken_agent3=[]
avg_timetaken_agent4=[]
trajectory_length1=[]
trajectory_length2=[]
trajectory_length3=[]
trajectory_length4=[]
avg_trajectory_length1=[]
avg_trajectory_length2=[]
avg_trajectory_length3=[]
avg_trajectory_length4=[]
number_of_bumps1=[]
number_of_bumps2=[]
number_of_bumps3=[]
number_of_bumps4=[]
avg_number_of_bumps1=[]
avg_number_of_bumps2=[]
avg_number_of_bumps3=[]
avg_number_of_bumps4=[]
pathlength_kg1=[]
pathlength_kg2=[]
pathlength_kg3=[]
pathlength_kg4=[]
avg_pathlength_kg1=[]
avg_pathlength_kg2=[]
avg_pathlength_kg3=[]
avg_pathlength_kg4=[]

dim=101
density=[0,0.03,0.06,0.09,0.12,0.15,0.18,0.21,0.24,0.27,0.3,0.33]
for p  in density:
    print("Density :"+ str(p))
    iteration=1
    while (iteration<=30):
        print("Iteration :"+ str(iteration))
        grid=creategrid(dim,p)
        start_time1= time.time()
        path1,kg1,bumps1=agent1.main(dim,"No",p,grid)
        if len(path1)==0:
            continue
        path2,kg2,bumps2=agent2.main(dim,"No",p,grid)
        endtime1=time.time() - start_time1
        timetaken_agent1.append(endtime1)
        trajectory_length1.append(len(path1))
        number_of_bumps1.append(bumps1)

        start_time2= time.time()
        path2,kg2,bumps2=agent2.main(dim,"No",p,grid)
        endtime2=time.time() - start_time2
        timetaken_agent2.append(endtime2)
        trajectory_length2.append(len(path2))
        number_of_bumps2.append(bumps2)

        start_time3= time.time()
        path3,kg3,bumps3=agent3.main(dim,"No",p,grid)
        endtime3=time.time() - start_time3
        timetaken_agent3.append(endtime3)
        trajectory_length3.append(len(path3))
        number_of_bumps3.append(bumps3)

        start_time4= time.time()
        path4,kg4,bumps4=agent4.main(dim,"No",p,grid)
        endtime4=time.time() - start_time4
        timetaken_agent4.append(endtime4)
        trajectory_length4.append(len(path4))
        number_of_bumps4.append(bumps4)

        

        

        #----Calculate shortest path in the KG-----##
        for i in range(0, dim):
            for j in range(0, dim):
                if(kg1[i][j] == 1):
                    kg1[i][j] = 0
        for i in range(0, dim):
            for j in range(0, dim):
                if(kg1[i][j] == 2):
                    kg1[i][j] = 1
        for i in range(0, dim):
            for j in range(0, dim):
                if(kg2[i][j] == 1):
                    kg2[i][j] = 0
        for i in range(0, dim):
            for j in range(0, dim):
                if(kg2[i][j] == 2):
                    kg2[i][j] = 1
        for i in range(0, dim):
            for j in range(0, dim):
                if(kg3[i][j] == -1):
                    kg3[i][j] = 0
        for i in range(0, dim):
            for j in range(0, dim):
                if(kg4[i][j] == -1):
                    kg4[i][j] = 0
        
        ll,path1_kg=astar.main(kg1,dim)
        pathlength_kg1.append(len(path1_kg))
        
        ll,path2_kg=astar.main(kg2,dim)
        pathlength_kg2.append(len(path2_kg))

        ll,path3_kg=astar.main(kg3,dim)
        pathlength_kg3.append(len(path3_kg))

        ll,path4_kg=astar.main(kg4,dim)
        pathlength_kg4.append(len(path4_kg))

        iteration+=1
        
        
    avg_timetaken_agent1.append(sum(timetaken_agent1)/len(timetaken_agent1))
    avg_timetaken_agent3.append(sum(timetaken_agent3)/len(timetaken_agent3))
    avg_timetaken_agent2.append(sum(timetaken_agent2)/len(timetaken_agent2))
    avg_timetaken_agent4.append(sum(timetaken_agent4)/len(timetaken_agent4))

    avg_trajectory_length1.append(sum(trajectory_length1)/len(trajectory_length1))
    avg_trajectory_length2.append(sum(trajectory_length2)/len(trajectory_length2))
    avg_trajectory_length4.append(sum(trajectory_length4)/len(trajectory_length4))
    avg_trajectory_length3.append(sum(trajectory_length3)/len(trajectory_length3))

    avg_number_of_bumps1.append(sum(number_of_bumps1)/len(number_of_bumps1))
    avg_number_of_bumps2.append(sum(number_of_bumps2)/len(number_of_bumps2))
    avg_number_of_bumps4.append(sum(number_of_bumps4)/len(number_of_bumps4))
    avg_number_of_bumps3.append(sum(number_of_bumps3)/len(number_of_bumps3))

    avg_pathlength_kg1.append(sum(pathlength_kg1)/len(pathlength_kg1))
    avg_pathlength_kg2.append(sum(pathlength_kg2)/len(pathlength_kg2))
    avg_pathlength_kg4.append(sum(pathlength_kg4)/len(pathlength_kg4))
    avg_pathlength_kg3.append(sum(pathlength_kg3)/len(pathlength_kg3))


plt.figure()
plt.plot(density, avg_timetaken_agent1,'-o')
plt.plot(density, avg_timetaken_agent2,'-o')
plt.plot(density, avg_timetaken_agent3,'-o')
plt.plot(density, avg_timetaken_agent4,'-o')
plt.legend(["AGENT1","AGENT2 ","AGENT3","AGENT4"])
plt.xticks(density)
plt.xlabel("Density")
plt.ylabel("Average Time taken")
plt.title("Density VS Average Time taken")
plt.savefig("graphs/Agent1-Agent2-Agent3-Agent4/DensityVSAverageTimetaken.png")
plt.close()

plt.figure()
plt.plot(density, avg_trajectory_length1,'-o')
plt.plot(density, avg_trajectory_length2,'-o')
plt.plot(density, avg_trajectory_length3,'-o')
plt.plot(density, avg_trajectory_length4,'-o')
plt.legend(["AGENT1","AGENT2 ","AGENT3","AGENT4"])
plt.xticks(density)
plt.xlabel("Density")
plt.ylabel("Average trajectory length")
plt.title("Density VS Average trajectory length")
plt.savefig("graphs/Agent1-Agent2-Agent3-Agent4/DensityVSAveragetrajectorylength.png")
plt.close()

plt.figure()
plt.plot(density, avg_number_of_bumps1,'-o')
plt.plot(density, avg_number_of_bumps2,'-o')
plt.plot(density, avg_number_of_bumps3,'-o')
plt.plot(density, avg_number_of_bumps4,'-o')
plt.legend(["AGENT1","AGENT2 ","AGENT3","AGENT4"])
plt.xticks(density)
plt.xlabel("Density")
plt.ylabel("Average number of bumps")
plt.title("Density VS Average number of bumps")
plt.savefig("graphs/Agent1-Agent2-Agent3-Agent4/DensityVSAveragenumberofbumps.png")
plt.close()

plt.figure()
plt.plot(density, avg_pathlength_kg1,'-o')
plt.plot(density, avg_pathlength_kg2,'-o')
plt.plot(density, avg_pathlength_kg3,'-o')
plt.plot(density, avg_pathlength_kg4,'-o')
plt.legend(["AGENT1","AGENT2 ","AGENT3","AGENT4"])
plt.xticks(density)
plt.xlabel("Density")
plt.ylabel("Average path length in knowledge grid")
plt.title("Density VS Average path length in knowledge grid")
plt.savefig("graphs/Agent1-Agent2-Agent3-Agent4/DensityVSAveragepathlengthinknowledgegrid.png")
plt.close()




