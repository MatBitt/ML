import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r', -1:'b'}
        if self.visualization :
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
 
    def train(self, data):
        self.data = data

        # { |w| : [w,b] }
        opt_dict = {}

        # Test all 4 symetrical points
        transforms = [[1,1],[1,-1],[-1,1],[-1,-1]]

        all_data = []

        # Gets all points
        for yi in self.data:
            for featureset in self.data[yi]:
                for features in featureset:
                    all_data.append(features)
        
        # Seeks for the highest and lowest feature in all points
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        
        #Releases data
        all_data = None

        # Starts with large steps, then gets more precise
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001
                      ]

        # Extremly expensive
        b_range_multiple = 5

        # b step multiplier
        # b doesn't need as much precision as w
        b_multiple = 5

        latest_optimum = self.max_feature_value*10

        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum])

            #Só pode fazer isso pq é convexo
            optimized = False
            while not optimized:
                #np.arrange(min, max, step)
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                  (self.max_feature_value*b_range_multiple),
                                   step*b_range_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        #yi(xi*w+b) >= 1 
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    found_option = False
                                    #Pode quebrar tudo a partir daqui
                        
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]
                
                if w[0] < 0 :
                    optimized = True
                    print('Optimized a step.')
                else:
                    # w = [5,5]
                    # step = 1
                    # w - step = [4,4]
                    w = w - step

            #Sorted the dictionary from lowest to highest |w|
            norms = sorted([n for n in opt_dict])
            
            #|w| : [w,b]
            opt_choice = opt_dict[norms[0]]

            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step*2
        
        for i in self.data:
            for xi in self.data[i]:
                yi=i
                print(xi,':', "%1.3f" %(yi*(np.dot(self.w,xi) + self.b)) )
                

       
    
    def predict(self, features):
        classification = np.sign(np.dot(np.array(features), self.w)+ self.b)
        if classification != 0 and self.visualization:
             self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
        
        return classification

    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # xw + b = v
        # x,y is an unknown point on the hyperplane
        # x_v and w_v are the vector
        # x_v = [x,y]
        # x_v.w_v+b = 1 for postive sv
        # x.w[0] + y.w[1] + b =1 
        # y = -x.w[0] - b + 1 / w[1]
        def hyperplane(x, w, b, v):
            return (-w[0]*x-b+v)/w[1]
        
        datarange = (self.min_feature_value*0.9, self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (wx + b) = 1
        # Positive support vector Hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min,hyp_x_max], [psv1, psv2])

        # (wx + b) = -1
        # Negative support vector Hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min,hyp_x_max], [nsv1, nsv2])
        
        # (wx + b) = 0
        # Decision Boundary
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min,hyp_x_max], [db1, db2])

        plt.show()



data_dict = {-1:np.array([[1,7],
                          [2,8],
                          [3,8]]),
              1:np.array([[5,1],
                          [6,-1],
                          [7,3]])}
                          
svm = Support_Vector_Machine()
svm.train(data=data_dict)

novos_dados = [[0,10],
               [1,3],
               [3,4],
               [3,5],
               [5,5],
               [5,6],
               [6,-5],
               [5,8]]

for ponto in novos_dados:
    svm.predict(ponto)

svm.visualize()