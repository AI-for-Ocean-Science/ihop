import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def exp_var(data_path, col_wise=False):
    data = np.load(data_path)
    
    # transpose data matrix if it is col_wise.
    if col_wise:
        data = np.transpose(data)

    # Calculate covariance matrix
    cov_data = np.cov(data)

    # Eigen decomposition 
    values, vectors = np.linalg.eig(cov_data)

    # Get explained variances
    explained_variances = values / np.sum(values)
    return explained_variances

if __name__ == "__main__":
    print("Explained Variance Computation Starts.")
    exp_var_list = []
    index_list = []
    for i in range(2, 11):
        data_path = f"../data/L23_NMF_bb_{i}_coef.npy"
        exp_var_i = exp_var(data_path)
        exp_var_list.append(exp_var_i)
        index_list.append(f"bb_{i}")
    result_dict = {
        "index_list": index_list,
        "exp_var": exp_var_list,
    }
    df_exp_var = pd.DataFrame(result_dict)
    df_exp_var.set_index("index_list", inplace=True)
    file_save = "../data/exp_var_coef_L23_NMF_bb.csv"
    df_exp_var.to_csv(file_save, header=False)
    exp_var_list = []
    index_list = []
    for i in range(2, 11):
        data_path = f"../data/L23_NMF_a_{i}_coef.npy"
        exp_var_i = exp_var(data_path)
        exp_var_list.append(exp_var_i)
        index_list.append(f"a_{i}")
    result_dict = {
        "index_list": index_list,
        "exp_var": exp_var_list,
    }
    df_exp_var = pd.DataFrame(result_dict)
    df_exp_var.set_index("index_list", inplace=True)
    file_save = "../data/exp_var_coef_L23_NMF_a.csv"
    df_exp_var.to_csv(file_save, header=False)    
    print("Computation Ends Successfully!")
    
    
    