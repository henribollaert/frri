#%%
import numpy as np
import warnings
from scipy.stats import gamma
import copy
import gurobipy as gb
import time
from sklearn.base import BaseEstimator
big_num = 1e7
from cvxopt import matrix, spmatrix, solvers
from mosek import iparam
import multiprocessing


#%%

def dom_relation(x, y, arg, orient=1):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    ext_dim_x = y.shape[0]
    ext_dim_y = x.shape[0]
    x_ext = np.repeat(x[:, np.newaxis, :], ext_dim_x, axis=1)
    y_ext = np.repeat(y[np.newaxis, :, :], ext_dim_y, axis=0)
    if orient == 1:
        res = np.maximum(np.minimum(1 - (y_ext - x_ext) / arg, 1), 0)
    else:
        res = np.maximum(np.minimum(1 - (x_ext - y_ext) / arg, 1), 0)
    return np.min(res, -1)

def triangular_dominance(x, y, arg=1, orient=1):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    ext_dim_x = y.shape[0]
    ext_dim_y = x.shape[0]
    x_ext = np.repeat(x[:, np.newaxis, :], ext_dim_x, axis=1)
    y_ext = np.repeat(y[np.newaxis, :, :], ext_dim_y, axis=0)
    if orient == 1:
        res = np.maximum(np.minimum(1 - (y_ext - x_ext) / arg, 1), 0)
    else:
        res = np.maximum(np.minimum(1 - (x_ext - y_ext) / arg, 1), 0)
    
    if res.shape[-1] > 0:
        return np.min(res, -1)
    else:
        return 1 + np.sum(res, -1)


###### TODO FIX GAUSSIAN DOMINANCE

def gaussian_dominance (x, y, arg=1, orient=1):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    ext_dim_x = y.shape[0]
    ext_dim_y = x.shape[0]
    x_ext = np.repeat(x[:, np.newaxis, :], ext_dim_x, axis=1)
    y_ext = np.repeat(y[np.newaxis, :, :], ext_dim_y, axis=0)
    if orient == 1:
        res = np.minimum(1*(y_ext >= x_ext) + np.exp(-(y_ext - x_ext)**2/arg),1)
        res = np.product(res, -1)
    else:
        res = np.minimum(1 * (x_ext >= y_ext) + np.exp(-(y_ext - x_ext)**2 / arg), 1)
        res = np.product(res, -1)
    return res


def quadratic_dominance(x, y, arg =1, orient=1):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    ext_dim_x = y.shape[0]
    ext_dim_y = x.shape[0]
    x_ext = np.repeat(x[:, np.newaxis, :], ext_dim_x, axis=1)
    y_ext = np.repeat(y[np.newaxis, :, :], ext_dim_y, axis=0)
    if orient == 1:
        res = np.minimum(np.min(1*(y_ext >= x_ext), -1) + np.maximum((1 - np.linalg.norm((y_ext - x_ext), axis=-1)/arg),0), 1)
        #res = np.maximum(np.minimum(1 - (y_ext - x_ext)**2 / arg, 1), 0)
    else:
        res = np.minimum(np.min(1*(x_ext >= y_ext), -1) + np.maximum((1 - np.linalg.norm((x_ext - y_ext), axis=-1)/arg),0), 1)
    return res


def manhattan_dominance(x, y, arg =1, orient=1):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    ext_dim_x = y.shape[0]
    ext_dim_y = x.shape[0]
    x_ext = np.repeat(x[:, np.newaxis, :], ext_dim_x, axis=1)
    y_ext = np.repeat(y[np.newaxis, :, :], ext_dim_y, axis=0)
    if orient == 1:
        res = np.minimum(np.min(1*(y_ext >= x_ext), -1) + np.maximum((1 - np.linalg.norm((y_ext - x_ext), axis=-1, ord=1)/arg),0), 1)
        #res = np.maximum(np.minimum(1 - (y_ext - x_ext)**2 / arg, 1), 0)
    else:
        res = np.minimum(np.min(1*(x_ext >= y_ext), -1) + np.maximum((1 - np.linalg.norm((x_ext - y_ext), axis=-1, ord=1)/arg),0), 1)
    return res

def manhattan_dominance(x, y, arg =1, orient=1):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    ext_dim_x = y.shape[0]
    ext_dim_y = x.shape[0]
    x_ext = np.repeat(x[:, np.newaxis, :], ext_dim_x, axis=1)
    y_ext = np.repeat(y[np.newaxis, :, :], ext_dim_y, axis=0)
    if orient == 1:
        res = np.minimum(np.min(1*(y_ext >= x_ext), -1) + np.maximum((1 - np.linalg.norm((y_ext - x_ext), axis=-1, ord=1)/arg),0), 1)
        #res = np.maximum(np.minimum(1 - (y_ext - x_ext)**2 / arg, 1), 0)
    else:
        res = np.minimum(np.min(1*(x_ext >= y_ext), -1) + np.maximum((1 - np.linalg.norm((x_ext - y_ext), axis=-1, ord=1)/arg),0), 1)
    return res

def manhattan_similarity (x, y, arg=1):
    return np.minimum(manhattan_dominance(x, y, arg, 1), manhattan_dominance(x, y, arg, -1))

def mahalanobis_dominance(x, y, sigma, arg=1, orient=1):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    ext_dim_x = y.shape[0]
    ext_dim_y = x.shape[0]
    x_ext = np.repeat(x[:, np.newaxis, :], ext_dim_x, axis=1)
    y_ext = np.repeat(y[np.newaxis, :, :], ext_dim_y, axis=0)
    L = np.linalg.cholesky(sigma)
    if orient == 1:
        res = np.minimum(np.min(1*(y_ext >= x_ext), -1) + np.maximum((1 - np.linalg.norm((y_ext - x_ext)@ L, axis=-1)/arg),0), 1)
        #res = np.maximum(np.minimum(1 - (y_ext - x_ext)**2 / arg, 1), 0)
    else:
        res = np.minimum(np.min(1*(x_ext >= y_ext), -1) + np.maximum((1 - np.linalg.norm((x_ext - y_ext)@ L, axis=-1)/arg),0), 1)
    return res

def mahalanobis_similarity(x, y, sigma, arg=1):
    return np.minimum(mahalanobis_dominance(x, y, sigma, arg, 1), mahalanobis_dominance(x, y, sigma, arg, -1))

def gaussian_similarity (x, y, arg=1):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    ext_dim_x = y.shape[0]
    ext_dim_y = x.shape[0]
    x_ext = np.repeat(x[:, np.newaxis, :], ext_dim_x, axis=1)
    y_ext = np.repeat(y[np.newaxis, :, :], ext_dim_y, axis=0)
    res = np.exp(-(y_ext - x_ext)**2/arg)
    res = np.product(res, -1)
    return res


def laplace_similarity (x, y, arg=1):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    ext_dim_x = y.shape[0]
    ext_dim_y = x.shape[0]
    x_ext = np.repeat(x[:, np.newaxis, :], ext_dim_x, axis=1)
    y_ext = np.repeat(y[np.newaxis, :, :], ext_dim_y, axis=0)
    res = np.exp(-np.abs(y_ext - x_ext)/arg)
    res = np.product(res, -1)
    return res

def quadratic_similarity (x, y, arg=1):
    return np.minimum(quadratic_dominance(x, y, arg, 1), quadratic_dominance(x, y, arg, -1))

def product_similarity(x, y):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    ext_dim_x = y.shape[0]
    ext_dim_y = x.shape[0]
    x_ext = np.repeat(x[:, np.newaxis, :], ext_dim_x, axis=1)
    y_ext = np.repeat(y[np.newaxis, :, :], ext_dim_y, axis=0)
    res = np.exp(np.log(np.minimum(x_ext, y_ext) + 1e-7) - np.log(np.maximum(x_ext, y_ext) + 1e-7))
    return np.min(res, -1)

def nilpotent_similarity(x, y):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    ext_dim_x = y.shape[0]
    ext_dim_y = x.shape[0]
    x_ext = np.repeat(x[:, np.newaxis, :], ext_dim_x, axis=1)
    y_ext = np.repeat(y[np.newaxis, :, :], ext_dim_y, axis=0)
    res = 1*(x_ext > y_ext) * np.maximum(1 - x_ext, y_ext) + 1*(x_ext < y_ext) * np.maximum(1 - y_ext, x_ext) + 1*(x_ext == y_ext)
    return np.min(res, -1)


def cosine_similarity2(x, y, arg=1):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    ext_dim_x = y.shape[0]
    ext_dim_y = x.shape[0]
    x_ext = np.repeat(x[:, np.newaxis, :], ext_dim_x, axis=1)
    y_ext = np.repeat(y[np.newaxis, :, :], ext_dim_y, axis=0)
    res = 0.5 * (1 + np.sum(x_ext*y_ext, axis=-1)/(np.linalg.norm(x_ext, axis=-1)*np.linalg.norm(y_ext, axis=-1)))
    res = np.maximum(1 - (1 - res)/arg, 0)
    return res


def cosine_similarity(x, y, arg=1):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    ext_dim_x = y.shape[0]
    ext_dim_y = x.shape[0]
    x_ext = np.repeat(x[:, np.newaxis, :], ext_dim_x, axis=1)
    y_ext = np.repeat(y[np.newaxis, :, :], ext_dim_y, axis=0)
    res = 0.5 * (1 + np.sum(x_ext*y_ext, axis=-1)/(np.linalg.norm(x_ext, axis=-1)*np.linalg.norm(y_ext, axis=-1)))
    res = np.maximum(1 - np.arccos(res)/(arg*np.pi/2), 0)
    return res


def discernibility_matrix (x,  y):
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    x_ext = np.repeat(x[:, np.newaxis], len(y), axis=1)
    y_ext = np.repeat(y[np.newaxis, :], len(x), axis=0)
    return 1*(x_ext == y_ext)


def nilpotent_t_norm(x, y):
    res = 1*(x + y > 1)
    res = res*np.minimum(x,y)
    return res

def nilpotent_implicator(x, y):
    res = 1*(x < y)
    res = res + np.maximum(1 - x, y)
    return np.minimum(res, 1)

def lukasiewicz_t_norm(x, y, degree=1.):
    if degree == 1.:
        return np.maximum(x + y - 1, 0)
    return np.power(np.maximum(np.power(x, degree) + np.power(y, degree) - 1, 0), 1/degree)

def lukasiewicz_t_conorm(x, y, degree=1.):
    if degree == 1.:
        return np.minimum(x + y, 1)
    return np.power(np.minimum(np.power(x, degree) + np.power(y, degree), 1), 1/degree)

def lukasiewicz_implicator(x, y, degree=1.):
    return np.power(np.minimum(1 - np.power(x, degree) + np.power(y, degree), 1), 1/degree)


def lukasiewicz_negator(x, degree=1.):
    return np.power(1 - np.power(x, degree), 1/degree)

def product_t_norm(x, y):
    return x*y

def product_implicator(x, y):
    return np.exp(np.log(np.minimum(x, y) + 1e-7) - np.log(x + 1e-7))

def godel_t_norm(x, y):
    return np.minimum(x, y)

def godel_implicator(x, y):
    return 1*(x < y) + y

def cosine_t_norm(x,y):
    return np.maximum(x*y - np.sqrt(1 - x*x)*np.sqrt(1-y*y), 0)


def custom_implicator(x, y):
    val = (2 - 2*x + y)/(2 - x)
    return np.minimum(val, 1)

def aux_func(x, y):
    if x <= y:
        return 1
    else:
        return x*y/(x - y + x*y)

'''
def custom_implicator2(x, y):
    aux_func = lambda a, b: 1 if a <= b else a*b/(a - b + a*b)
    vec_func = np.vectorize(aux_func)
    return vec_func(x, y)

def custom_implicator3(x, y):
    aux_func = lambda a, b: 1 if a <= b else a*b/((np.sqrt(a) - np.sqrt(b) + np.sqrt(a*b))**2)
    vec_func = np.vectorize(aux_func)
    return vec_func(x, y)
'''

def custom_implicator2(x, y):
    return np.exp(np.log(x*y + 1e-7) - np.log(np.maximum(x - y, 0) + x*y + 1e-7))

def custom_implicator3(x, y):
    return np.exp(np.log(x*y + 1e-7) - 2*np.log(np.maximum(np.sqrt(x) - np.sqrt(y), 0) + np.sqrt(x*y) + 1e-7))

def ind_relation(x, y, arg=1.):
    return np.minimum(dom_relation(x, y, arg, 1), dom_relation(x, y, arg, -1))

def triangular_similarity(x, y, arg=1.):
    return np.minimum(triangular_dominance(x, y, arg, 1), triangular_dominance(x, y, arg, -1))





def max_ind_relation(x, y, arg=1., attribute_set=None):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    n_attributes = x.shape[1]
    if attribute_set is None:
        attribute_set = np.arange(n_attributes)
    if attribute_set.size == 0:
        return np.ones((x.shape[0], y.shape[0]))
    res = np.minimum(triangular_dominance(x[:, attribute_set], y[:, attribute_set], arg, orient=1),
                     triangular_dominance(x[:, attribute_set], y[:, attribute_set], arg, orient=-1))

    return np.squeeze(res)


def general_triangular_relation(x, y, arg, individual_types):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    dom1_indices_c = np.where(individual_types == 1)[0]
    dom1_indices_cc = np.where(individual_types == -1)[0]
    ind_indeces = np.where(individual_types == 2)[0]
    comparisons = []
    if dom1_indices_c.size > 0:
        dom_c = triangular_dominance(x[:,dom1_indices_c], y[:,dom1_indices_c], arg)
        comparisons.append(dom_c)
    if dom1_indices_cc.size > 0:
        dom_cc = triangular_dominance(x[:,dom1_indices_cc], y[:,dom1_indices_cc], arg, -1)
        comparisons.append(dom_cc)
    if ind_indeces.size > 0:
        ind = triangular_similarity(x[:,ind_indeces], y[:,ind_indeces], arg)
        comparisons.append(ind)
    if len(comparisons) == 0:
        return np.ones((x.shape[0], y.shape[0]))
    return np.min(np.concatenate(comparisons, -1), -1)


def get_dom_low_apr(relation_matrix, fuzzy_set, implicator=lukasiewicz_implicator, upward=True):
    relation_matrix = np.atleast_2d(relation_matrix)
    if relation_matrix.shape[0] != relation_matrix.shape[1]:
        warnings.warn("Relation matrix is not a square matrix", Warning)
    if upward:
        low_apr = np.min(implicator(relation_matrix.T, fuzzy_set), axis=1)
    else:
        low_apr = np.min(implicator(relation_matrix, fuzzy_set), axis=1)
    return low_apr

def get_dom_upp_apr(relation_matrix, fuzzy_set, t_norm=lukasiewicz_t_norm, upward=True):
    relation_matrix = np.atleast_2d(relation_matrix)
    if relation_matrix.shape[0] != relation_matrix.shape[1]:
        warnings.warn("Relation matrix is not a square matrix", Warning)
    if upward:
        upp_apr = np.max(t_norm(relation_matrix, fuzzy_set), axis=1)
    else:
        upp_apr = np.max(t_norm(relation_matrix.T, fuzzy_set), axis=1)
    return upp_apr


def get_owa_dom_low_apr(relation_matrix, fuzzy_set, weights, implicator=lukasiewicz_implicator, upward=True):
    relation_matrix = np.atleast_2d(relation_matrix)
    if relation_matrix.shape[0] != relation_matrix.shape[1]:
        warnings.warn("Relation matrix is not a square matrix", Warning)
    n_samples = relation_matrix.shape[0]
    if relation_matrix.shape[0] != relation_matrix.shape[1]:
        warnings.warn("Relation matrix is not a square matrix", Warning)
    if weights == 'add':
        weights = get_add_weights(n_samples)
    elif weights == 'exp':
        weights = get_exp_weights(n_samples)
    elif weights == 'invadd':
        weights = get_invadd_weights(n_samples)
    
    if upward:
        low_apr = owa_min(implicator(relation_matrix.T, fuzzy_set), weights, 1)
    else:
        low_apr = owa_min(implicator(relation_matrix, fuzzy_set), weights, 1)
    return low_apr

def get_owa_dom_upp_apr(relation_matrix, fuzzy_set, weights, t_norm=lukasiewicz_t_norm, upward=True):
    relation_matrix = np.atleast_2d(relation_matrix)
    if relation_matrix.shape[0] != relation_matrix.shape[1]:
        warnings.warn("Relation matrix is not a square matrix", Warning)
    n_samples = relation_matrix.shape[0]
    if relation_matrix.shape[0] != relation_matrix.shape[1]:
        warnings.warn("Relation matrix is not a square matrix", Warning)
    if weights == 'add':
        weights = get_add_weights(n_samples)
    elif weights == 'exp':
        weights = get_exp_weights(n_samples)
    elif weights == 'invadd':
        weights = get_invadd_weights(n_samples)

    if upward:
        upp_apr = owa_max(t_norm(relation_matrix, fuzzy_set), weights, 1)
    else:
        upp_apr = owa_max(t_norm(relation_matrix.T, fuzzy_set), weights, 1)
    return upp_apr


def get_ind_low_apr(relation_matrix, fuzzy_set, implicator=lukasiewicz_implicator):
    relation_matrix = np.atleast_2d(relation_matrix)
    if relation_matrix.shape[0] != relation_matrix.shape[1]:
        warnings.warn("Relation matrix is not a square matrix", Warning)
    low_apr = np.min(implicator(relation_matrix, fuzzy_set), 1)
    return low_apr

'''
def get_ind_low_apr2(X, fuzzy_set, arg=1.):
    indiscernibility_values = triangular_similarity(X, X, arg)
    low_apr = np.min(1 - indiscernibility_values + fuzzy_set, 1)
    min_indexes = np.argmin(1 - indiscernibility_values + fuzzy_set, 1)
    return low_apr, min_indexes
'''


def get_ind_upp_apr(relation_matrix, fuzzy_set, t_norm=lukasiewicz_t_norm):
    relation_matrix = np.atleast_2d(relation_matrix)
    if relation_matrix.shape[0] != relation_matrix.shape[1]:
        warnings.warn("Relation matrix is not a square matrix", Warning)
    upp_apr = np.max(t_norm(relation_matrix, fuzzy_set), 1)
    return upp_apr
 

'''
def get_ind_apr(relation_matrix, fuzzy_set, t_norm=lukasiewicz_t_norm,  implicator=lukasiewicz_implicator, lbda=.5):
    relation_matrix = np.atleast_2d(relation_matrix)
    if relation_matrix.shape[0] != relation_matrix.shape[1]:
        warnings.warn("Relation matrix is not a square matrix", Warning)
    low_apr = get_ind_low_apr(relation_matrix, fuzzy_set, implicator=implicator)
    upp_apr = get_ind_upp_apr(relation_matrix, fuzzy_set, t_norm=t_norm)
    return lbda*upp_apr + (1 - lbda)*low_apr


def get_ind_apr2(X, fuzzy_set, arg=1.):
    labels = np.sort(list(set(fuzzy_set)))
    labels = labels.astype(int)
    indeces = []
    for label in labels:
        indeces.append(np.where(fuzzy_set == label)[0])

    n_samples = X.shape[0]
    n_classes = len(labels)
    ind_relations = ind_relation(X, X, arg)
    discernibility = 1 * (fuzzy_set.reshape(-1, 1) != fuzzy_set)
    similarities = discernibility * ind_relations
    closest_points = np.argmax(similarities, 1)
    # %%
    closest_lists = []
    examined_points = np.ones(n_samples) == 0
    close_pairs = set()
    for i in range(n_samples):
        if not examined_points[i]:
            current_list = []
            tmp = i
            while True:
                current_list.append(tmp)
                examined_points[tmp] = True
                tmp = closest_points[tmp]
                if closest_points[tmp] == current_list[-1]:
                    current_list.append(tmp)
                    examined_points[tmp] = True
                    close_pairs.add((current_list[-1], current_list[-2]))
                    break
            closest_lists.append(current_list)
    indeces2 = []
    for label in labels:
        indeces2.append(set())

    indeces3 = []
    for label in labels:
        indeces3.append(set(indeces[label]))

    pos_region = -np.ones(n_samples)

    for pair in close_pairs:
        val = (2 - similarities[pair[0]][pair[1]]) / 2
        for label in labels:
            if pair[0] in indeces3[label]:
                indeces2[label].add(pair[0])
                indeces3[label].remove(pair[0])
            if pair[1] in indeces3[label]:
                indeces2[label].add(pair[1])
                indeces3[label].remove(pair[1])
        pos_region[pair[0]] = val
        pos_region[pair[1]] = val

    labels2 = set(labels)
    while len(labels2) > 0:
        label = np.random.choice(list(labels2))
        index = np.random.choice(list(indeces3[label]))
        other_indeces = set()
        for labb in labels:
            if labb != label:
                other_indeces = other_indeces.union(indeces2[labb])
        tmp = list(other_indeces)
        pos_region[index] = np.min(lukasiewicz_implicator(pos_region[tmp],lukasiewicz_negator(ind_relation(X[index], X[tmp], arg))))
        indeces3[label].remove(index)
        indeces2[label].add(index)
        if len(indeces3[label]) == 0:
            labels2.remove(label)

    approximations = np.zeros((n_classes, n_samples))
    for label in labels:
        approximations[label][indeces[label]] = pos_region[indeces[label]]
        approximations[label] = get_ind_upp_apr(X, approximations[label])
    return approximations[1]



def get_ind_apr3(relation_matrix, fuzzy_set, t_norm=lukasiewicz_t_norm, implicator=lukasiewicz_implicator,
                 lbda=.5, ind=0):
    relation_matrix = np.atleast_2d(relation_matrix)
    if relation_matrix.shape[0] != relation_matrix.shape[1]:
        warnings.warn("Relation matrix is not a square matrix", Warning)
    n_samples = relation_matrix.shape[0]
    approx = get_ind_apr(relation_matrix, fuzzy_set, t_norm=t_norm, implicator=implicator , lbda=lbda)
    #approx = get_ind_upp_apr(relation_matrix, approx, t_norm=t_norm)
    run = True
    br = 0
    previous_unlignes_indeces = np.arange(n_samples)
    while run:
        #print(approx[ind], " ", fuzzy_set[ind])
        br = br+1
        run = False
        indices = np.arange(n_samples)
        aligned = 0
        unligned_indeces = []
        for i in indices:
            mask = np.ones(n_samples, bool)
            mask[i] = False
            if approx[i] > fuzzy_set[i] + 1e-6:
                tmp = np.max(t_norm(relation_matrix[i][mask], approx[mask]))
                if np.abs(tmp - approx[i]) > 1e-6:
                    unligned_indeces.append(i)
                    run = True
                else:
                    aligned = aligned + 1
                approx[i] = tmp
            elif approx[i] < fuzzy_set[i] - 1e-6:
                tmp = np.min(implicator(relation_matrix[i][mask], approx[mask]))
                if np.abs(tmp - approx[i]) > 1e-6:
                    run = True
                    unligned_indeces.append(i)
                else:
                    aligned = aligned + 1
                approx[i] = tmp
        unligned_indeces = np.array(unligned_indeces)
        if len(previous_unlignes_indeces) == len(unligned_indeces):
            for i in unligned_indeces:
                approx[i] = fuzzy_set[i]
        previous_unlignes_indeces = copy.deepcopy(unligned_indeces)

    return approx

def get_ind_apr5(init_set, relation_matrix, fuzzy_set, t_norm=lukasiewicz_t_norm, implicator=lukasiewicz_implicator):
    relation_matrix = np.atleast_2d(relation_matrix)
    if relation_matrix.shape[0] != relation_matrix.shape[1]:
        warnings.warn("Relation matrix is not a square matrix", Warning)
    n_samples = relation_matrix.shape[0]
    approx = copy.deepcopy(init_set)
    run = True
    args = np.zeros(n_samples, dtype=int)
    while run:
        run = False
        indices = np.arange(n_samples)
        for i in indices:
            mask = np.ones(n_samples, bool)
            mask[i] = False
            if approx[i] > fuzzy_set[i]:
                tmp = np.max(t_norm(relation_matrix[i][mask], approx[mask]))
                args[i] = np.argmax(t_norm(relation_matrix[i][mask], approx[mask]))
                if np.abs(tmp - approx[i]) > 1e-6:
                    run = True
                approx[i] = tmp
            elif approx[i] < fuzzy_set[i]:
                tmp = np.min(implicator(relation_matrix[i][mask], approx[mask]))
                args[i] = np.argmin(t_norm(relation_matrix[i][mask], approx[mask]))
                if np.abs(tmp - approx[i]) > 1e-6:
                    run = True
                approx[i] = tmp

    return approx, args


def get_general_granular_approx(init_set, relation_matrix_x, relation_matrix_y, t_norm=lukasiewicz_t_norm, implicator=lukasiewicz_implicator):
    relation_matrix_x = np.atleast_2d(relation_matrix_x)
    relation_matrix_y = np.atleast_2d(relation_matrix_y)
    if relation_matrix_x.shape[0] != relation_matrix_x.shape[1]:
        warnings.warn("Relation matrix on predictors is not a square matrix", Warning)
    if relation_matrix_y.shape[0] != relation_matrix_y.shape[1]:
        warnings.warn("Relation matrix on predictions is not a square matrix", Warning)
    if relation_matrix_x.shape[0] != relation_matrix_y.shape[0] or relation_matrix_x.shape[1] != relation_matrix_y.shape[1]:
        warnings.warn("matrices have to be of the same size", Warning)
    implications = implicator(relation_matrix_x, relation_matrix_y)
    n_samples = relation_matrix_x.shape[0]
    approx = copy.deepcopy(init_set)
    run = True
    #args = np.zeros(n_samples, dtype=int)
    while run:
        
        run = False
        indices = np.arange(n_samples)
        
        matrix_for_mins = implicator(approx_old, implications)
        #matrix_for_mins = matrix_for_mins[~np.eye(matrix_for_mins.shape[0],dtype=bool)].reshape(matrix_for_mins.shape[0], -1)
        approx_new = np.min(matrix_for_mins, axis=-1)
        args = np.argmin(matrix_for_mins, axis=-1)
        margin = np.max(np.abs(approx_new - approx_old))
        print(margin)
        time.sleep(.5)
        if margin < 1e-6:
            break
        approx_old = copy.deepcopy(approx_new)x§
        for i in range(n_samples):
            tmp = np.min(implicator(approx, implications[i]))
            if np.abs(tmp - approx[i]) > 1e-6:
                run = True
            approx[i] = tmp
    return approx
    
    '''

def get_general_granular_approx10(init_set, relation_matrix_x, relation_matrix_y, learning_rate = 0.05, t_norm=lukasiewicz_t_norm, implicator=lukasiewicz_implicator):
    relation_matrix_x = np.atleast_2d(relation_matrix_x)
    relation_matrix_y = np.atleast_2d(relation_matrix_y)
    if relation_matrix_x.shape[0] != relation_matrix_x.shape[1]:
        warnings.warn("Relation matrix on predictors is not a square matrix", Warning)
    if relation_matrix_y.shape[0] != relation_matrix_y.shape[1]:
        warnings.warn("Relation matrix on predictions is not a square matrix", Warning)
    if relation_matrix_x.shape[0] != relation_matrix_y.shape[0] or relation_matrix_x.shape[1] != relation_matrix_y.shape[1]:
        warnings.warn("matrices have to be of the same size", Warning)
    implications = implicator(relation_matrix_x, relation_matrix_y)
    n_samples = relation_matrix_x.shape[0]
    approx_old = copy.deepcopy(init_set)
    run = True
    #args = np.zeros(n_samples, dtype=int)
    iter_num = 0
    while True:
        iter_num = iter_num + 1
        matrix_for_mins = implicator(approx_old, implications)
        #matrix_for_mins = matrix_for_mins[~np.eye(matrix_for_mins.shape[0],dtype=bool)].reshape(matrix_for_mins.shape[0], -1)
        approx_new = approx_old - learning_rate*(approx_old - np.min(matrix_for_mins, axis=-1))
        margin = np.max(np.abs(approx_new - approx_old))
        #print(margin)
        #time.sleep(.5)
        if margin < 1e-15:
            print(iter_num)
            return approx_new
        approx_old = copy.deepcopy(approx_new)

        '''


def get_general_granular_approx11(init_set, relation_matrix_x, relation_matrix_y, owa_weights, learning_rate = 0.05, t_norm=lukasiewicz_t_norm, implicator=lukasiewicz_implicator):
    relation_matrix_x = np.atleast_2d(relation_matrix_x)
    relation_matrix_y = np.atleast_2d(relation_matrix_y)
    if relation_matrix_x.shape[0] != relation_matrix_x.shape[1]:
        warnings.warn("Relation matrix on predictors is not a square matrix", Warning)
    if relation_matrix_y.shape[0] != relation_matrix_y.shape[1]:
        warnings.warn("Relation matrix on predictions is not a square matrix", Warning)
    if relation_matrix_x.shape[0] != relation_matrix_y.shape[0] or relation_matrix_x.shape[1] != relation_matrix_y.shape[1]:
        warnings.warn("matrices have to be of the same size", Warning)
    implications = implicator(relation_matrix_x, relation_matrix_y)
    n_samples = relation_matrix_x.shape[0]
    approx_old = copy.deepcopy(init_set)
    run = True
    #args = np.zeros(n_samples, dtype=int)
    iter_num = 0
    while True:
        iter_num = iter_num + 1
        matrix_for_mins = implicator(approx_old, implications)
        #matrix_for_mins = matrix_for_mins[~np.eye(matrix_for_mins.shape[0],dtype=bool)].reshape(matrix_for_mins.shape[0], -1)
        approx_new = approx_old - learning_rate*(approx_old - owa_min(matrix_for_mins, owa_weights, axis=1))
        margin = np.max(np.abs(approx_new - approx_old))
        #print(margin)
        #time.sleep(.5)
        if margin < 1e-6:
            print(iter_num)
            return approx_new
        approx_old = copy.deepcopy(approx_new)


def get_general_granular_approx12(init_set, relation_matrix_x, relation_matrix_y, learning_rate = 0.05, t_norm=lukasiewicz_t_norm, implicator=lukasiewicz_implicator, t_conorm = lukasiewicz_t_conorm):
    relation_matrix_x = np.atleast_2d(relation_matrix_x)
    relation_matrix_y = np.atleast_2d(relation_matrix_y)
    if relation_matrix_x.shape[0] != relation_matrix_x.shape[1]:
        warnings.warn("Relation matrix on predictors is not a square matrix", Warning)
    if relation_matrix_y.shape[0] != relation_matrix_y.shape[1]:
        warnings.warn("Relation matrix on predictions is not a square matrix", Warning)
    if relation_matrix_x.shape[0] != relation_matrix_y.shape[0] or relation_matrix_x.shape[1] != relation_matrix_y.shape[1]:
        warnings.warn("matrices have to be of the same size", Warning)
    implications = implicator(relation_matrix_x, relation_matrix_y)
    n_samples = relation_matrix_x.shape[0]
    approx_old = copy.deepcopy(init_set)
    run = True
    #args = np.zeros(n_samples, dtype=int)
    iter_num = 0
    while True:
        iter_num = iter_num + 1
        matrix_for_mins = implicator(approx_old, implications)
        #matrix_for_mins = matrix_for_mins[~np.eye(matrix_for_mins.shape[0],dtype=bool)].reshape(matrix_for_mins.shape[0], -1)
        tmp1 = t_norm(approx_old, 1- learning_rate)
        tmp1 = approx_old*(1 - learning_rate)
        tmp2 = t_norm(np.min(matrix_for_mins, axis=-1), learning_rate)
        tmp2 = np.min(matrix_for_mins, axis=-1)* learning_rate
        approx_new = t_conorm(tmp1, tmp2)
        approx_new = tmp1 + tmp2
        margin = np.max(np.abs(approx_new - approx_old))
        #print(tmp1)
        #time.sleep(1)
        
        if margin < 1e-15:
            print(iter_num)
            return approx_new
        approx_old = copy.deepcopy(approx_new)
        
    

def get_general_granular_approx2(init_set, relation_matrix_x, relation_matrix_y, learning_rate = 0.1): 
    n_samples = len(init_set)
    old_aprox = copy.deepcopy(init_set)
    implications = lukasiewicz_implicator(relation_matrix_x, relation_matrix_y)
    while True:
        new_approx = copy.deepcopy(old_aprox)
        new_approx = new_approx - 2*learning_rate*(new_approx - np.mean(new_approx))/n_samples
        matrix_for_mins = lukasiewicz_implicator(new_approx, implications)
        new_approx = np.min(matrix_for_mins, axis=-1)
        if np.max(np.abs(new_approx - old_aprox)) < 1e-6:
            return new_approx
        old_aprox = copy.deepcopy(new_approx)


'''


def get_granular_approximation_quantile_lukasiewicz(rel_matrix, fuzzy_set, p=.5):
    if p == 1.:
        p = p - 1e-6
    elif p == 0.:
        p = p + 1e-6
    model = gb.Model("granular_approximation")
    model.modelSense = gb.GRB.MINIMIZE
    x = []
    y = []
    alphas = []
    n_samples = rel_matrix.shape[0]
    for i in range(n_samples):
        x.append(model.addVar(vtype=gb.GRB.CONTINUOUS, lb=0, obj=p))
        y.append(model.addVar(vtype=gb.GRB.CONTINUOUS, lb=0, obj=1 - p))
        alphas.append(model.addVar(vtype=gb.GRB.CONTINUOUS, lb=-float('inf'), obj= 0))
    
    for i in range(n_samples):
        for j in range(n_samples): 
            model.addConstr(alphas[i] - alphas[j] + 1 >= rel_matrix[i][j])  
    
    for i in range(n_samples):
        model.addConstr(fuzzy_set[i] - alphas[i] == x[i] - y[i])
    

    model.setParam("OutputFlag", 0)
    model.optimize()

    lambdas = np.zeros(n_samples)
    
    for i in range(n_samples):
        lambdas[i] = alphas[i].x

    return lambdas


def get_granular_approximation_quantile_product(rel_matrix, fuzzy_set, p=.5):
    model = gb.Model("granular_approximation")
    model.modelSense = gb.GRB.MINIMIZE
    x = []
    y = []
    alphas = []
    n_samples = rel_matrix.shape[0]
    
    for i in range(n_samples):
        x.append(model.addVar(vtype=gb.GRB.CONTINUOUS, lb=0, ub=1, obj=p))
        y.append(model.addVar(vtype=gb.GRB.CONTINUOUS, lb=0, ub=1, obj=1 - p))
        alphas.append(model.addVar(vtype=gb.GRB.CONTINUOUS, lb=0, obj=0))
    
    for i in range(n_samples):
        for j in range(n_samples):
            model.addConstr(alphas[i] >= rel_matrix[i][j]*alphas[j])
    
    for i in range(n_samples):
        model.addConstr(fuzzy_set[i] - alphas[i] == x[i] - y[i])
    

    model.setParam("OutputFlag", 0)
    model.optimize()

    lambdas = np.zeros(n_samples)
    
    for i in range(n_samples):
        lambdas[i] = alphas[i].x

    return lambdas


def get_granular_approximation_expectation_lukasiewicz (rel_matrix, fuzzy_set):
    model = gb.Model("granular_approximation")
    model.modelSense = gb.GRB.MINIMIZE
    n_samples = rel_matrix.shape[0]
    alphas = []
    for i in range(n_samples):
        alphas.append(model.addVar(vtype=gb.GRB.CONTINUOUS, lb=0, ub=1, obj=0))
    
    #print(nn_approx)
    for i in range(n_samples):
        for j in range(n_samples):
            model.addConstr(alphas[i] - alphas[j] + 1 >= rel_matrix[i][j])   
    
    model.setObjective(gb.quicksum((alphas[i] - fuzzy_set[i])**2 for i in range(n_samples)))
    
    model.setParam("OutputFlag", 0)
    model.optimize()

    lambdas = np.zeros(n_samples)
    
    for i in range(n_samples):
        lambdas[i] = alphas[i].x

    return lambdas


def get_granular_approximation_expectation_product (rel_matrix, fuzzy_set):
    model = gb.Model("granular_approximation")
    model.modelSense = gb.GRB.MINIMIZE
    n_samples = rel_matrix.shape[0]
    alphas = []
    for i in range(n_samples):
        alphas.append(model.addVar(vtype=gb.GRB.CONTINUOUS, lb=0, ub=1, obj=0))
    
    for i in range(n_samples):
        for j in range(n_samples):  
            model.addConstr(alphas[i] >= rel_matrix[i][j]*alphas[j])
    
    model.setObjective(gb.quicksum((alphas[i] - fuzzy_set[i])**2 for i in range(n_samples)))

    model.setParam("OutputFlag", 0)
    model.optimize()

    lambdas = np.zeros(n_samples)
    
    for i in range(n_samples):
        lambdas[i] = alphas[i].x

    return lambdas



def get_multiclass_granular_approx_mae(relation_matrix_x, relation_matrix_y, weights=None, nn_approx=-1): 
    
    n_samples = relation_matrix_x.shape[0]
    if weights is None:
        weights = np.ones(n_samples)
    if nn_approx == -1:
        nn_approx = int(n_samples)
    model = gb.Model("relation_granular_approximation")
    model.modelSense = gb.GRB.MAXIMIZE
    implications = lukasiewicz_implicator(relation_matrix_x, relation_matrix_y)
    lambdas = []
    for i in range(n_samples):
        lambdas.append(model.addVar(vtype=gb.GRB.CONTINUOUS, lb=0, ub = 1, obj=weights[i]))
    

    for i in range(n_samples):
        arr = -(1 - relation_matrix_y[i])*relation_matrix_x[i]
        inds = np.argpartition(arr, kth=nn_approx-1)[:nn_approx]
        for j in inds:
            model.addConstr(lambdas[i] <= 1 - lambdas[j] + implications[i][j])
    
    #for i in range(n_samples):
     #   model.addConstr(min_val <= weights[i]*lambdas[i])
    
    model.setParam("OutputFlag", 0)
    model.optimize()

    lambdas_opt = np.zeros(n_samples)
    for i in range(n_samples):
        lambdas_opt[i] = lambdas[i].x
    
    return lambdas_opt


def get_multiclass_granular_approx_mse(relation_matrix_x, relation_matrix_y, weights=None, nn_approx=-1): 
    
    n_samples = relation_matrix_x.shape[0]
    if weights is None:
        weights = np.ones(n_samples)
    if nn_approx == -1:
        nn_approx = int(n_samples)
    model = gb.Model("relation_granular_approximation")
    model.modelSense = gb.GRB.MINIMIZE
    implications = lukasiewicz_implicator(relation_matrix_x, relation_matrix_y)
    lambdas = []
    for i in range(n_samples):
        lambdas.append(model.addVar(vtype=gb.GRB.CONTINUOUS, lb=0, ub = 1, obj=0))
    
    
    for i in range(n_samples):
        arr = -(1 - relation_matrix_y[i])*relation_matrix_x[i]
        inds = np.argpartition(arr, kth=nn_approx-1)[:nn_approx]
        for j in inds:
            model.addConstr(lambdas[i] <= 1 - lambdas[j] + implications[i][j])
    
    
    model.setObjective(gb.quicksum(weights[i]*(lambdas[i] - 1)**2 for i in range(n_samples)))

    model.setParam("OutputFlag", 0)
    model.setParam("Threads", 0)
    model.optimize()

    lambdas_opt = np.zeros(n_samples)
    for i in range(n_samples):
        lambdas_opt[i] = lambdas[i].x
    
    return lambdas_opt


def get_multiclass_granular_approx_mae_cvxopt(relation_matrix_x, relation_matrix_y, weights=None, nn_approx=-1, n_jobs=None): 
    
    n_samples = relation_matrix_x.shape[0]
    if weights is None:
        weights = np.ones(n_samples)
    if nn_approx == -1:
        nn_approx = int(n_samples)

    implications = lukasiewicz_implicator(relation_matrix_x, relation_matrix_y)
    
    q = matrix(-weights)


    inter_class_relation_x = (1 - relation_matrix_y)*relation_matrix_x
    tmp_important = inter_class_relation_x > 0
    closest_inds = np.argpartition(-inter_class_relation_x, kth=nn_approx-1, axis=1)[:,:nn_approx]
    if_bigger_zero = np.take_along_axis(tmp_important, closest_inds, axis=1)
    n_closest = np.sum(if_bigger_zero, axis=1)
    tmp_conc1 = np.repeat(np.arange(n_samples), n_closest)
    tmp_conc2 = closest_inds[if_bigger_zero]
    matrix_inds = np.unique(np.sort(np.concatenate([tmp_conc1.reshape(-1,1), tmp_conc2.reshape(-1, 1)], axis=1), axis=1), axis=0)

    num_comb = matrix_inds.shape[0]
    first_index_constr = np.repeat(np.arange(num_comb), 2)
    first_index_bound = np.arange(num_comb, num_comb + n_samples)
    #mesh = np.meshgrid(range(n_samples), range(n_samples))
    #mesh_bool = (mesh[1] > mesh[0]).flatten()
    #tmp = np.concatenate([mesh[1].reshape(-1,1), mesh[0].reshape(-1,1)], axis=1)
    second_index_constr = matrix_inds.flatten().astype(int)
    second_index_bound = np.arange(n_samples)
    values_constr= np.ones(2*num_comb)
    values_bound = np.ones(n_samples)
    G = spmatrix(np.concatenate((values_constr, values_bound)),
                np.concatenate((first_index_constr, first_index_bound)),
                np.concatenate((second_index_constr, second_index_bound)))
    right_side_constr = implications[matrix_inds[:,0], matrix_inds[:,1]] + 1
    right_side_bound = np.ones(n_samples)
    h = matrix(np.concatenate((right_side_constr, right_side_bound)))
    
    num_threads = 1
    if n_jobs == None:
        num_threads = 1
    elif n_jobs == -1:
        num_threads = 0
    elif n_jobs == 0:
        warnings.warn("Invalid value. 'n_jobs=1' is used.")
        num_threads = 1
    elif n_jobs > 0:
        num_threads = n_jobs
    elif n_jobs <-1:
        total_threads = multiprocessing.cpu_count()
        num_threads = total_threads + n_jobs + 1
    

    solvers.options['mosek'] = {iparam.log: 0, iparam.num_threads: num_threads}
    #solvers.options['mosek'] = {iparam.log: 0, iparam.num_threads: 0}
    sol=solvers.lp(q, G, h,solver='mosek')
    lambdas_opt = np.array(sol['x']).flatten()
    return lambdas_opt


def get_multiclass_granular_approx_mse_cvxopt(relation_matrix_x, relation_matrix_y, weights=None, nn_approx=-1, n_jobs=None): 
    
    n_samples = relation_matrix_x.shape[0]
    if weights is None:
        weights = np.ones(n_samples)
    if nn_approx == -1:
        nn_approx = int(n_samples)

    implications = lukasiewicz_implicator(relation_matrix_x, relation_matrix_y)
    
    P = spmatrix(weights, range(n_samples), range(n_samples))
    q = matrix(-weights)

    inter_class_relation_x = (1 - relation_matrix_y)*relation_matrix_x
    tmp_important = inter_class_relation_x > 0
    closest_inds = np.argpartition(-inter_class_relation_x, kth=nn_approx-1, axis=1)[:,:nn_approx]
    if_bigger_zero = np.take_along_axis(tmp_important, closest_inds, axis=1)
    n_closest = np.sum(if_bigger_zero, axis=1)
    tmp_conc1 = np.repeat(np.arange(n_samples), n_closest)
    tmp_conc2 = closest_inds[if_bigger_zero]
    matrix_inds = np.unique(np.sort(np.concatenate([tmp_conc1.reshape(-1,1), tmp_conc2.reshape(-1, 1)], axis=1), axis=1), axis=0)

    num_comb = matrix_inds.shape[0]
    first_index = np.repeat(np.arange(num_comb), 2)
    first_index_bound = np.arange(num_comb, num_comb + n_samples)
    second_index = matrix_inds.flatten().astype(int)
    second_index_bound = np.arange(n_samples)
    values = np.ones(2*num_comb)
    values_bound = np.ones(n_samples)
    #G = spmatrix(values, first_index, second_index)
    G = spmatrix(np.concatenate((values, values_bound)),
                np.concatenate((first_index, first_index_bound)),
                np.concatenate((second_index, second_index_bound)))
    #print(G.size)
    right_side = implications[matrix_inds[:,0], matrix_inds[:,1]] + 1
    right_side_bound = np.ones(n_samples)
    h = matrix(np.concatenate((right_side, right_side_bound)))
    #h = matrix(implications[matrix_inds[:,0], matrix_inds[:,1]] + 1)
    

    num_threads = 1
    if n_jobs == None:
        num_threads = 1
    elif n_jobs == -1:
        num_threads = 0
    elif n_jobs == 0:
        warnings.warn("Invalid value. 'n_jobs=1' is used.")
        num_threads = 1
    elif n_jobs > 0:
        num_threads = n_jobs
    elif n_jobs <-1:
        total_threads = multiprocessing.cpu_count()
        num_threads = total_threads + n_jobs + 1
    

    solvers.options['mosek'] = {iparam.log: 0, iparam.num_threads: num_threads}
    #solvers.options['mosek'] = {iparam.log: 0, iparam.num_threads: 0}
    sol=solvers.qp(P, q, G, h,solver='mosek')
    lambdas_opt = np.array(sol['x']).flatten()
    return lambdas_opt






def get_owa_ind_low_apr(relation_matrix, fuzzy_set, weights, implicator=lukasiewicz_implicator, arg=1.):
    relation_matrix = np.atleast_2d(relation_matrix)
    n_samples = relation_matrix.shape[0]
    if relation_matrix.shape[0] != relation_matrix.shape[1]:
        warnings.warn("Relation matrix is not a square matrix", Warning)
    if weights == 'add':
        weights = get_add_weights(n_samples)
    elif weights == 'exp':
        weights = get_exp_weights(n_samples)
    elif weights == 'invadd':
        weights = get_invadd_weights(n_samples)
    low_apr = owa_min(implicator(relation_matrix, fuzzy_set), weights, 1)
    return low_apr


def get_owa_ind_upp_apr(relation_matrix, fuzzy_set, weights, t_norm=lukasiewicz_t_norm, arg=1.):
    relation_matrix = np.atleast_2d(relation_matrix)
    n_samples = relation_matrix.shape[0]
    if relation_matrix.shape[0] != relation_matrix.shape[1]:
        warnings.warn("Relation matrix is not a square matrix", Warning)
    if weights == 'add':
        weights = get_add_weights(n_samples)
    elif weights == 'exp':
        weights = get_exp_weights(n_samples)
    elif weights == 'invadd':
        weights = get_invadd_weights(n_samples)
    low_apr = owa_max(t_norm(relation_matrix, fuzzy_set), weights, 1)
    return low_apr


def get_owa_ind_apr(relation_matrix, fuzzy_set, weights, t_norm=lukasiewicz_t_norm, implicator=lukasiewicz_implicator, lbda=.5, arg=1.):
    relation_matrix = np.atleast_2d(relation_matrix)
    if relation_matrix.shape[0] != relation_matrix.shape[1]:
        warnings.warn("Relation matrix is not a square matrix", Warning)
    low_apr = get_owa_ind_low_apr(relation_matrix, fuzzy_set, weights, implicator=implicator ,arg=arg)
    upp_apr = get_owa_ind_upp_apr(relation_matrix, fuzzy_set, weights, t_norm=t_norm, arg=arg)
    return lbda * upp_apr + (1 - lbda) * low_apr

def get_add_weights(n):
    arr = np.flip(np.arange(n) + 1)
    return arr/np.sum(arr)

def get_exp_weights(n):
    arr = 2**(-np.arange(n, dtype='float') - 1)
    return arr/np.sum(arr)


def get_invadd_weights(n):
    arr = np.flip(np.sort(1/(1 + np.arange(n))))
    return arr/np.sum(arr)


def get_gamma_weights(n, gamma_shape, gamma_scale):
    generator = gamma(gamma_shape, 0, gamma_scale).pdf
    weights = generator(.01 + np.arange(n) / n * 10)
    return weights/np.sum(weights)
    

def owa_min(array, weighs, axis=0):
    array = np.atleast_2d(array)
    if axis == 1:
        array = np.sort(array)
    elif axis == 0:
        array = np.sort(array.T)
    weighs = weighs/np.sum(weighs)

    return np.squeeze(np.dot(array, weighs))

def owa_add_min(array, axis=0):
    array = np.array(array)
    if axis == 0:
        weights = get_add_weights(array.shape[0])
    elif axis == 1:
        weights = get_add_weights(array.shape[1])
    return owa_min(array, weights, axis)

def owa_exp_min(array, axis=0):
    array = np.array(array)
    if axis == 0:
        weights = get_exp_weights(array.shape[0])
    elif axis == 1: 
        weights = get_exp_weights(array.shape[1])
    return owa_min(array, weights, axis)

def owa_invadd_min(array, axis=0):
    array = np.array(array)
    if axis == 0:
        weights = get_invadd_weights(array.shape[0])
    elif axis == 1:
        weights = get_invadd_weights(array.shape[1])
    return owa_min(array, weights, axis)

def owa_max(array, weighs, axis=0):
    array = np.atleast_2d(array)
    if axis == 1:
        array = np.flip(np.sort(array), 1)
    elif axis == 0:
        array = np.flip(np.sort(array.T), 1)
    weighs = weighs/np.sum(weighs)

    return np.squeeze(np.dot(array, weighs))


def owa_add_max(array, axis=0):
    array = np.array(array)
    if axis == 0:
        weights = get_add_weights(array.shape[0])
    elif axis == 1:
        weights = get_add_weights(array.shape[1])
    return owa_max(array, weights, axis)

def owa_exp_max(array, axis=0):
    array = np.array(array)
    if axis == 0:
        weights = get_exp_weights(array.shape[0])
    elif axis == 1:
        weights = get_exp_weights(array.shape[1])
    return owa_max(array, weights, axis)

def owa_invadd_max(array, axis=0):
    array = np.array(array)
    if axis == 0:
        weights = get_invadd_weights(array.shape[0])
    elif axis == 1:
        weights = get_invadd_weights(array.shape[1])
    return owa_max(array, weights, axis)


def is_subset(set1, set2, tol=0.0001):
    set1 = np.array(set1)
    set2 = np.array(set2)
    res = np.sum(1 * (set2 < set1 - tol))
    if res == 0:
        return True
    else:
        return False
# %%
