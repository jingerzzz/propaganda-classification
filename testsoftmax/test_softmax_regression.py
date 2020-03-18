import numpy as np
import matplotlib.pyplot as plt


# Load data
q2_data = np.load('q2_data/q2_data.npz')
q2x_train = q2_data['q2x_train']
q2y_train = q2_data['q2y_train']
q2x_test = q2_data['q2x_test']
q2y_test = q2_data['q2y_test']

def h_w(w,x):## x is dxN, w is dxk, return kxN
    return w.T*x

def softmax(z): ## z is kxN, return kxN probablity matrix
    return np.exp(z) / np.sum(np.exp(z), axis=0)


def get_labelization_dict(y):
    y = set(y)
    y = list(y)
    label_to_name = {}
    name_to_label = {}
    for i in range(len(y)):
        label_to_name[i]=y[i]
        name_to_label[y[i]]=i
    return (label_to_name,name_to_label)
    
def labelize(y,name_to_label):
    y_label = []
    for i in range(len(y)):
        y_label.append(name_to_label[y[i]])
    return y_label

def delabelize(y,label_to_name):
    y_name = []
    for i in range(len(y)):
        y_name.append(label_to_name[y[i]])
    return y_name

    
    
def pred_label(softmax_z):
    y_pred = softmax_z.argmax(axis=0)
    return y_pred



def loglikelihood(softmax_z,y):## soft_max is kxN, y is 1xN (y has been processed so names have been tranformed to labels)
    l = 0
    for j in range(softmax_z.shape[1]):
        l+=np.log(softmax_z[y[0,j],j])
    return l
    
def init_w(x,k):## y is non duplicate list
    d = x.shape[0]
    w = np.zeros((d,k))
    w = np.asmatrix(w)
    w[:,-1]=0
    return w
    
def gradient(x,y,w):## x is dxN, w is dxk, y is 1xN, softmax_z is kxN,return dxk
    ## this function makes sure the last w, w_K, doens't change and always is zero.
    h = h_w(w,x)

    softmax_z = softmax(h)   
    k = w.shape[1]
    N = y.shape[1]
    y_I = np.zeros((k,N))## y_I is kxN
    for j in range(N):
        y_I[y[0,j],j]=1
    
    g = x*(y_I-softmax_z).T
    g[:,-1]=0
    return g
    
    
    
def gradient_ascent(x,y,rate=0.0005):
    k = y.max() + 1
    print(k)
    w = init_w(x,k)
    h = h_w(w,x)   
    print("h shape:{}".format(h.shape))
    softmax_z = softmax(h)
    print("softmax shape:{}".format(softmax_z.shape))
    l = loglikelihood(softmax_z,y)        
    g = gradient(x,y,w)
    
    stop_delta = 1e-5
    max_iteration = 1000
    iter_count =0
    delta = np.Infinity
    
    while abs(delta)>stop_delta and iter_count<max_iteration:
#         print("w:{}".format(w))
#         print("h:{}".format(h))
#         print("softmax:{}".format(softmax_z))
        # print("gradient:{}".format(g))
        # print("iter:{}, loglikelihood:{}".format(iter_count,l))
        iter_count+=1
        g = gradient(x,y,w)
        w = w + rate*g
        h = h_w(w,x)
        softmax_z = softmax(h)
        l_new = loglikelihood(softmax_z,y)
        delta = l_new - l
        l = l_new

    return w

def evaluate(y_pred, y_test):
    total = len(y_pred)
    correct = 0
    for i in range(total):
        if y_pred[i]==y_test[i]:
            correct+=1
    accuracy = correct/total
    return accuracy

def train_and_evaluate(x_train,y_train):
    x=np.asmatrix(q2x_train)
    x=x.T
    y=q2y_train
    y=y.tolist()
    for i in range(len(y)):
        y[i]=y[i][0]
    label_to_name, name_to_label = get_labelization_dict(y)
    y = labelize(y,name_to_label)

    y=np.asmatrix(y)

    w = gradient_ascent(x,y)

    x_test=np.asmatrix(q2x_test)
    x_test=x_test.T
    y_test = []
    for i in range(len(q2y_test)):
        y_test.append(q2y_test[i][0])
    # print(y_test)
    h = h_w(w,x_test)
    # print(h.shape)
    softmax_z = softmax(h)
    # print(softmax_z.shape)
    y_pred = pred_label(softmax_z)

    y_pred= np.asarray(y_pred)
    y_pred=y_pred.tolist()[0]
    y_pred = delabelize(y_pred,label_to_name)
    print("the accuracy is:{}".format(evaluate(y_pred,y_test)))

train_and_evaluate(q2x_train,q2y_train)