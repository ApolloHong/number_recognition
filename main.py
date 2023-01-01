import numpy as np


def read_txt(x):
    with open(x,'r') as file:
        M=[]
        for line in file:
            data=line.strip(' ').split(',')
            M.append([float(i) for i in data])
        return np.array(M)

# data = xlrd.open_workbook("C:\\Users\\xiaohong\\Desktop\\流形学习项目\\numbers_OCR_project.csv")
# table = data.sheets()
M = read_txt("C:\\Users\\xiaohong\\Desktop\\流形学习项目\\numbers_OCR_project.csv")


def sort_by_number():
    '''

    :return: 10 nums and its data
    '''
    lst = [[] for i in range(10)]
    nums = M[:,-1]
    c = 0
    for k in nums:
        vec = M[c, :-1]
        lst[int(k)].append(vec.tolist())
        c += 1
    return lst


def mean_and_variance(f):
    '''

    :param f: num(features) of vector
    :return: m is the maean of the sample
             Sigma is the variance matrice
    '''
    # m = []
    # num_features = f.shape[1]
    # for i in range(num_features):
    #     m.append(sum(f[:,i])/num_features)
    m = np.mean(f,axis=0)
    sigmas = np.var(f,axis=0)
    Sigma = np.diag(np.array(sigmas))
    return m , Sigma


def Rayleigh_ratio(x , m1 , m2 , Sigma1 , Sigma2):
    # R is Rayleigh_ratio
    R = (x.T @ m1-x.T @ m2)**2/(x.T @ Sigma1 @ x) + (x.T @ Sigma2 @ x)
    return R


def LDA_vector_maximize_Rayleigh_ratio(m1 , m2 , Sigma1 , Sigma2):
    M = Sigma1 + Sigma2
    v = np.linalg.pinv(M) @ (m1-m2)
    c = Rayleigh_ratio(v , m1 , m2 , Sigma1 , Sigma2)
    return v , c


def LDA(v , c , f_input , population1 , population2):
    if v.T @ f_input > c :
        print(f'this is {population1}')
    else :
        print(f'this  is {population2}')


def binary_search():
    for i in range(10):
        for j in range(i+1,10):
            lst = sort_by_number()
            f1,f2 = np.array(lst[i]),np.array(lst[j])
            m1,Sigma1 = mean_and_variance(f1)
            m2,Sigma2 = mean_and_variance(f2)
            v ,c = LDA_vector_maximize_Rayleigh_ratio(m1 , m2 , Sigma1 , Sigma2)
            LDA(v,c,f,i,j)

binary_search()







