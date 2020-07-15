import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
print("successful")



#def initialize_parameter_deep(layer_dims):
    #parameters = {}
    #L = len(layer_dims)
    #for l in range(1,L):
        #parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
        #parameters['b' + str(l)] = np.zeros((layer_dims[l],1))


    #return parameters
#parameters = initialize_parameter_deep([5,4,3])
#print("W1 = " + str(parameters["W1"]))
#print("b1 = " + str(parameters["b1"]))
#print("W2 = " + str(parameters["W2"]))
#print("b2 = " + str(parameters["b2"]))


def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s
print("sigmoid(0) = ", sigmoid(0))
print("sigmoid(9.2) = ", sigmoid(9.2))
    
def  initialize_parameter(dim):
    w = np.zeros((dim,1))
    b = 0

    return w,b

dim = 2
w, b = initialize_parameter(dim)
print("w = "+str(w))
print("b = "+str(b))

def propagate(w,b,X,Y):
    m = X.shape[1]
    
    #FORWARD PROPAGATION
    A = sigmoid(np.dot(w.T,X)+b)

    cost = (-1/m)*np.sum(Y*np.log(A)+(1-Y)*(np.log(1-A)))

    #BACKWARD PROPAGATION
    dw = (1/m)*np.dot(X,(A-Y).T)
    db = (1/m)*np.sum(A-Y)

    cost = np.squeeze(cost)
    grads = {"dw":dw,
             "db":db}
    return grads,cost

w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))

def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost=False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X,Y)
        dw = grads['dw']
        db = grads['db']
        w = w - learning_rate*dw
        b = b - learning_rate*db

        if i%100 == 0:
            costs.append(cost)
        #print the cost every 100 training examples
        if print_cost and i%100 == 0:
            print("cost after iteration %i : %f " %(i, cost))

    params = {'w':w,
              'b':b}
    grads = {"dw":dw,
             "db":db}
    return params, grads, costs


params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)

print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
            
def predict(w,b,X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)

    A = sigmoid(np.dot(w.T,X)+b)

    for i in range(A.shape[1]):
        Y_prediction[0,i]= 1 if A[0,i]>0.5 else 0

    return Y_prediction

print("predictions = "+str(predict(w,b,X)))

data = pd.read_csv("breast-cancer-wisconsin.data")
print(data.head(5))
data.columns = ['id','Clump Thickness','Unicellsize','Unicellshape','Margin Adhesion','EpithelialcellSize','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']

print(data.head(5))
data.drop(['id'], inplace= True, axis = 1)

data.to_csv("data.csv", index= None, header= True)
data1 = pd.read_csv("data.csv")
data1.replace(to_replace='?', value=data.mean())

def retx(x):
    if x == 2:
        return 0
    else:
        return 1

data['Class'] = data['Class'].map(retx)
data['Class'] = data['Class'].map(lambda x : 0 if x == 2 else 1)
print(data['Class'].head(5))
    



x = np.array(data1.drop(['Class'], axis=1))

print(np.shape(x))
y = np.array(data1['Class'])
print(np.shape(y))


print(x)


print(np.shape(y))
print("sanity check after reshaping : "+ str(x[0:5,100]))


[X_train, X_test,Y_train, Y_test] = train_test_split(x,y,test_size=0.1,random_state=0)
print(X_test)
X_train = X_train.reshape(9, 628)
Y_train = Y_train.reshape(1,628)
Y_test = Y_test.reshape(1,70)
X_test = X_test.reshape(9, 70)
print(X_test)
print(Y_test)


def model(X_train, Y_train, X_test, Y_test, num_iterations= 1000,learning_rate=0.5,print_cost= False):
    w, b = initialize_parameter(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train,num_iterations,learning_rate, print_cost)
    w = parameters['w']
    b = parameters['b']
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    print("train accuracy : {}%".format(100- np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

d = model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations(per hundreds)')
plt.title('Learning rate = ' +str(d["learning_rate"]))
print(plt.show())

learning_rates = [0.01,0.001, 0.0001]
models = {}
for i in learning_rates:
          print("learning rate is : " +str(i))
          models[str(i)]= model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
          print('\n' + "______________________________________________________________")

for i in learning_rates:
          plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]['learning_rates']))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center',shadow= True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
print(plt.show())
                                                                                 
          

    
