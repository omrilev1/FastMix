import os
import hashlib
import numpy as np
import pandas as pd 
import urllib 
import scipy
import glob
import re
from scipy.special import expit
from scipy.fft import dct, idct
from utils.fastwht.hadamard import *
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm 
from scipy.optimize import minimize_scalar
from time import perf_counter_ns
from sklearn.datasets import fetch_openml
import numpy as np
import math 
from time import perf_counter

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any

import sklearn.datasets


def filesystem_safe_output_path(path, *, create_parent=True):
    path = os.path.abspath(os.fspath(path))

    if os.name == "nt":
        path = os.path.normpath(path)
        if not path.startswith("\\\\?\\"):
            if path.startswith("\\\\"):
                path = "\\\\?\\UNC\\" + path.lstrip("\\")
            else:
                path = "\\\\?\\" + path

    if create_parent:
        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    return path

def compact_filename_token(tokens, *, prefix="", none_token="outer", hash_len=8):
    if isinstance(tokens, (str, bytes, os.PathLike)):
        normalized_tokens = [os.fspath(tokens)]
    else:
        normalized_tokens = [
            none_token if token is None else str(token)
            for token in tokens
        ]

    if len(normalized_tokens) == 0:
        raise ValueError("tokens must contain at least one value")

    if len(normalized_tokens) == 1:
        token = normalized_tokens[0]
    else:
        digest = hashlib.sha1("|".join(normalized_tokens).encode("utf-8")).hexdigest()[:hash_len]
        token = f"{len(normalized_tokens)}x_{normalized_tokens[0]}_{normalized_tokens[-1]}_{digest}"

    token = re.sub(r"[^A-Za-z0-9._-]+", "-", token).strip("-")
    return f"{prefix}{token}" if prefix else token

# Dataset-dependent suptitle
dataset_titles = {
    'california'          : "California Housing"  ,
    'beijing'             : "Beijing"             ,
    'years'               : "Years"               ,
    'synthetic_correlated': "Correlated"          ,
    'synthetic_eigenvalue': "Max eigenvalue"      ,
    'black_friday'        : "black_friday"        ,
    'superconduct'        : "Superconduct"        ,
    'airline'             : "Airline"             ,
    'rossman'             : "Rossman"             ,
    'nyc-taxi-green-dec-2016': "NYC Taxi Green Dec 2016",
    'nyc_taxi_green_dec_2016': "NYC Taxi Green Dec 2016"
}# n x d, with Q^T Q = I_d

# Default dataset used by scripts when they are configured with one entry.
SINGLE_DATASET_NAME = "beijing"

def _haar_orthogonal(d):
    A = np.random.randn(d, d)
    Q, R = np.linalg.qr(A)
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    Q = Q * s
    return Q  # d x d, with Q^T Q = I_d

def _sample_unit_sphere_rows(num_rows, d):
    X = np.random.randn(num_rows, d)
    row_norms = np.linalg.norm(X, axis=1, keepdims=True)
    row_norms[row_norms == 0.0] = 1.0
    return X / row_norms

def clip_row_norms(X_in, max_norm=1.0, eps=1e-12):
    """
    Clip (scale down) each row of X_in so its L2 norm is at most max_norm.
    Rows with norm <= max_norm are unchanged.
    """
    X = np.asarray(X_in, dtype=float)
    row_norms = np.linalg.norm(X, axis=1)
    scales = np.minimum(1.0, max_norm / (row_norms + eps))
    return X * scales[:, None]

################# Analytic solver for the noise required for the GaussMix mechanism #################

def objective_func_full_composition(alpha, k, gamma, delta, inflation_norm, T):
    if alpha <= 1 or alpha >= gamma/(1 + inflation_norm):
        return np.inf  # Penalize out-of-bound values

    term1 = (k * alpha) / (2 * (alpha - 1)) * np.log(1.0 - (1.0 + inflation_norm) / (gamma))
    term2 = - (k / (2 * (alpha - 1))) * np.log(1 - (alpha*(1 + inflation_norm)) / gamma)
    term3 = (np.log(1.0 / delta) + (alpha - 1)*np.log(1-1/alpha) - np.log(alpha)) / (alpha - 1)
    term4 = np.sqrt(2.0 * np.log(1.25/delta)) / (gamma/np.sqrt(k))
    return T * (term1 + term2) + term3 + term4

def objective_func_full_composition_fast(alpha, k, gamma, delta, T, inflation_norm, add_eigval=True):
    c = (1.0 + inflation_norm) / gamma
    z = alpha * c
    if alpha <= 1.0:
        return np.inf
    if 1.0 - c - 0.25 * c * c <= 0.0:
        return np.inf
    if 1.0 - z - 0.25 * z * z <= 0.0:
        return np.inf
    
    term1 = k * (alpha/ (2.0 * (alpha - 1.0))) * np.log(1.0 - (1.0 + inflation_norm) /gamma - (1/4) * ((1.0 + inflation_norm) /gamma)**2) \
                - k /(2.0*(alpha - 1.0)) * np.log(1.0 - alpha * (1.0 + inflation_norm) /gamma - (1/4) * (alpha * (1.0 + inflation_norm) /gamma)**2)
    term2 = (np.log(1.0 / delta) + (alpha - 1)*np.log(1-1/alpha) - np.log(alpha)) / (alpha - 1)
    term3 = np.sqrt(2.0 * np.log(1.25/delta)) / (gamma/np.sqrt(k))
    
    if add_eigval:
        return term1 * T + term2 + term3
    else:
        return term1 * T + term2

def objective_func_full(alpha, k, gamma, delta, inflation_norm):
    beta = (1.0 + inflation_norm) / gamma
    if alpha <= 1.0 or alpha >= 1.0 / beta:
        return np.inf
    if beta >= 1.0:
        return np.inf

    term1 = (k * alpha) / (2.0 * (alpha - 1.0)) * np.log1p(-beta)
    term2 = -(k / (2.0 * (alpha - 1.0))) * np.log1p(-alpha * beta)
    term3 = (np.log(1.0 / delta) + (alpha - 1.0) * np.log(1.0 - 1.0 / alpha) - np.log(alpha)) / (alpha - 1.0)
    term4 = np.sqrt(2.0 * np.log(1.25 / delta)) * np.sqrt(k) / gamma
    return term1 + term2 + term3 + term4

def objective_func_full_fast(alpha, k, gamma, delta, inflation_norm, add_eigval=True):
    c = (1.0 + inflation_norm) / gamma
    z = alpha * c

    if alpha <= 1.0:
        return np.inf
    if 1.0 - c - 0.25 * c * c <= 0.0:
        return np.inf
    if 1.0 - z - 0.25 * z * z <= 0.0:
        return np.inf

    term1 = k * (alpha / (2.0 * (alpha - 1.0))) * np.log1p(-c - 0.25 * c * c)
    term2 = -(k / (2.0 * (alpha - 1.0))) * np.log1p(-z - 0.25 * z * z)
    term3 = (np.log(1.0 / delta) + (alpha - 1.0) * np.log(1.0 - 1.0 / alpha) - np.log(alpha)) / (alpha - 1.0)
    term4 = np.sqrt(2.0 * np.log(1.25 / delta)) * np.sqrt(k) / gamma
    
    if add_eigval:
        return term1 + term2 + term3 + term4
    else:
        return term1 + term2 + term3

# Solve the required noise for composition of T sketching steps by 
# performing the full conversion from RenyiDP to DP numerically
def solve_gamma_renyi_full_composition(init_gamma, k, target_delta, target_epsilon, inflation_norm, T):   
    # Solve for the \gamma required for target \eps,\delta DP after composing T steps 
    # Define binary search bounds
    left, right = init_gamma / 500000.0, 500000.0*init_gamma
    best_gamma = right  # Default to upper bound in case no solution is found
    
    # The bound is for composition over 3 contributions for \delta.
    # Thus, we divide \delta by 3 
    internal_target_delta = target_delta/3.0
    while right - left > 1e-6:  # Precision threshold
        mid_gamma = (left + right) / 2
        if mid_gamma < 1 + 1e-5:
            return 1.0 + 1e-5
        else: 
            # Solve for optimal alpha given the current gamma
            result = scipy.optimize.minimize_scalar(objective_func_full_composition, 
                                     bounds=(1 + 1e-5, mid_gamma - 1e-5), 
                                     args=(k, mid_gamma, internal_target_delta, inflation_norm, T), 
                                     method='bounded') # minimization is between 1 < \alpha < \gamma 
        
        if result.success and result.fun < target_epsilon:
            best_gamma = mid_gamma  # Update best found gamma
            right = mid_gamma  # Search for a smaller gamma
        else:
            left = mid_gamma  # Increase gamma to meet target_epsilon
    return best_gamma

# Solve the required noise for composition of T sketching steps by 
# performing the full conversion from RenyiDP to DP numerically
def solve_gamma_renyi_full_fast_composition(init_gamma, k, target_delta, target_epsilon, T, inflation_norm, add_eigval=True):   
    # Solve for the \gamma required for target \eps,\delta DP after composing T steps 
    # Define binary search bounds
    left, right = init_gamma / 500000.0, 500000.0*init_gamma
    best_gamma = right  # Default to upper bound in case no solution is found
    
    # The bound is for composition over 3 contributions for \delta.
    # Thus, we divide \delta by 3 
    if add_eigval:
        internal_target_delta = target_delta/3.0
    else:
        internal_target_delta = target_delta 
    
    while right - left > 1e-6:  # Precision threshold
        mid_gamma = (left + right) / 2
        # Solve for optimal alpha given the current gamma
        if mid_gamma < 1 + 1e-5:
            return 1.0 + 1e-5
        else:
            result = scipy.optimize.minimize_scalar(objective_func_full_composition_fast, 
                                bounds=(1.0 + 1e-5, 4.0*mid_gamma/5.0), 
                                args=(k, mid_gamma, internal_target_delta, T, inflation_norm, add_eigval), 
                                method='bounded') # minimization is between 1 < \alpha
        
        if result.success and result.fun < target_epsilon:
            best_gamma = mid_gamma  # Update best found gamma
            right = mid_gamma  # Search for a smaller gamma
        else:
            left = mid_gamma  # Increase gamma to meet target_epsilon
    return best_gamma

############## Solve gamma full in both the Gaussian and in the fast cases ##############
def solve_gamma_renyi_full(init_gamma, k, target_delta, target_epsilon, inflation_norm):
    alpha_lo = 1.0 + 1e-5

    # Need gamma/(1+inflation_norm) > alpha_lo
    gamma_floor = (1.0 + inflation_norm) * alpha_lo * (1.0 + 1e-8)

    left = max(init_gamma / 500000.0, gamma_floor)
    right = max(500000.0 * init_gamma, left * 1.01)
    best_gamma = right

    internal_target_delta = target_delta / 3.0

    while right - left > 1e-6:
        mid_gamma = 0.5 * (left + right)

        alpha_hi = mid_gamma / (1.0 + inflation_norm) - 1e-5
        if alpha_hi <= alpha_lo:
            left = mid_gamma
            continue

        result = scipy.optimize.minimize_scalar(
            objective_func_full,
            bounds=(alpha_lo, alpha_hi),
            args=(k, mid_gamma, internal_target_delta, inflation_norm),
            method='bounded'
        )

        if result.success and np.isfinite(result.fun) and result.fun < target_epsilon:
            best_gamma = mid_gamma
            right = mid_gamma
        else:
            left = mid_gamma

    return best_gamma

def solve_gamma_renyi_full_fast(init_gamma, k, target_delta, target_epsilon, inflation_norm, add_eigval=True):
    alpha_lo = 1.0 + 1e-5
    c = 2.0 * (np.sqrt(2.0) - 1.0)

    # Need c * gamma/(1+inflation_norm) > alpha_lo
    gamma_floor = ((1.0 + inflation_norm) * alpha_lo / c) * (1.0 + 1e-8)

    left = max(init_gamma / 500000.0, gamma_floor)
    right = max(500000.0 * init_gamma, left * 1.01)
    best_gamma = right

    if add_eigval:
        internal_target_delta = target_delta / 3.0
    else:
        internal_target_delta = target_delta 
        
    while right - left > 1e-6:
        mid_gamma = 0.5 * (left + right)

        alpha_hi = c * mid_gamma / (1.0 + inflation_norm) - 1e-5
        if alpha_hi <= alpha_lo:
            left = mid_gamma
            continue

        result = scipy.optimize.minimize_scalar(
            objective_func_full_fast,
            bounds=(alpha_lo, alpha_hi),
            args=(k, mid_gamma, internal_target_delta, inflation_norm,add_eigval),
            method='bounded'
        )

        if result.success and np.isfinite(result.fun) and result.fun < target_epsilon:
            best_gamma = mid_gamma
            right = mid_gamma
        else:
            left = mid_gamma

    return best_gamma

####################### Generate Synthetic Data #######################
# Generate train-only synthetic dataset for OLS simulations.
def generate_synthetic_ols(n, d, type='correlated'):

    def finalize_dataset(X, y, theta_true):
        X_train = np.asarray(X, dtype=np.float64)
        y_train = np.asarray(y, dtype=np.float64)

        norm_fact_y = np.max(np.abs(y_train))
        if norm_fact_y > 0:
            y_train = y_train / norm_fact_y

        C_max = 1.0

        row_norm_max = np.sqrt(np.max(np.sum(X_train**2, axis=1)))
        if row_norm_max > 0:
            X_train = X_train / row_norm_max

        lambda_max = np.max(np.linalg.eigvalsh(X_train.T @ X_train))
        lambda_min = np.min(np.linalg.eigvalsh(X_train.T @ X_train))
        y_col = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
        XY = np.hstack((X_train, y_col))
        lambda_min_XY = np.real(np.min(np.linalg.eigvalsh(XY.T @ XY)))
        lambda_max_XY = np.real(np.max(np.linalg.eigvalsh(XY.T @ XY)))

        return C_max, X_train, y_train, lambda_min, lambda_max, lambda_min_XY, lambda_max_XY, theta_true

    distribution = str(type)
    mean = np.zeros(d)

    if distribution in {'Gaussian', 'synthetic_Gaussian'}:
        cov = np.eye(d)
        
    elif distribution in {'correlated', 'synthetic_correlated'}:
        idx = np.arange(d)
        cov = 2.0 * ((0.99) ** np.abs(idx[:, None] - idx[None, :]))  
        
    elif distribution in {'large_min_eigval'}:
        # Sample rows from the unit sphere so X^T X stays well-conditioned.
        X = _sample_unit_sphere_rows(n, d)
        
    else:
        raise ValueError(f"Unknown synthetic OLS dataset type: {type}")

    if distribution != 'large_min_eigval':
        X = np.random.multivariate_normal(mean=mean, cov=cov, size=n)
    noise = np.sqrt(0.1) * np.random.randn(n)

    # generate a sparse regressor
    theta_true = (1./np.sqrt(d))*np.random.randn(d,)
    y = X @ theta_true + noise

    return finalize_dataset(X, y, theta_true)

def _to_numeric_binary(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = series.astype(str).str.strip()
    numeric = pd.to_numeric(s, errors="coerce")
    mapping = {
        "1": 1.0, "0": 0.0,
        "yes": 1.0, "no": 0.0,
        "true": 1.0, "false": 0.0,
        "y": 1.0, "n": 0.0,
        "\u6709": 1.0, "\u65e0": 0.0,
        "\u662f": 1.0, "\u5426": 0.0,
    }
    mapped = s.str.lower().map(mapping)

    out = mapped.copy()
    missing = out.isna()
    out.loc[missing] = numeric.loc[missing]
    return pd.to_numeric(out, errors="coerce")

def _finalize_train_only_dataset(
    X_train,
    y_train,
    dataset_name,
    *,
    C_max=1.0,
    normalize_y=True,
    shuffle=True,
):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"Feature/target row mismatch for {dataset_name}: "
            f"{X_train.shape[0]} vs {y_train.shape[0]}"
        )

    if shuffle:
        permutation = np.random.permutation(X_train.shape[0])
        X_train = X_train[permutation, :]
        y_train = y_train[permutation]

    if normalize_y:
        norm_fact_y = np.max(np.abs(y_train))
        if norm_fact_y > 0:
            y_train = y_train / norm_fact_y

    norm_fact = np.sqrt(np.max(np.sum(X_train**2, axis=1)))
    if norm_fact > 0:
        X_train = X_train / norm_fact

    n, d = X_train.shape
    lambda_max = np.max(np.linalg.eigvalsh(X_train.T @ X_train))
    lambda_min = np.min(np.linalg.eigvalsh(X_train.T @ X_train))
    y_col = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
    XY = np.hstack((X_train, y_col))
    lambda_min_XY = np.real(np.min(np.linalg.eigvalsh(XY.T @ XY)))
    lambda_max_XY = np.real(np.max(np.linalg.eigvalsh(XY.T @ XY)))
    dataset_title = dataset_titles.get(dataset_name, str(dataset_name))

    return (
        float(C_max),
        X_train,
        y_train,
        np.real(lambda_min),
        np.real(lambda_max),
        np.real(lambda_min_XY),
        np.real(lambda_max_XY),
        n,
        d,
        dataset_title,
    )

def GetDataset(dataset_name=SINGLE_DATASET_NAME):
    if dataset_name is None:
        dataset_name = SINGLE_DATASET_NAME

    current_dir = os.getcwd()
    dataset_path = os.path.join(current_dir, "datasets")

    if dataset_name == 'years':
        data = np.loadtxt(
            os.path.join(dataset_path, 'YearPredictionMSD.txt'),
            delimiter=",",
            dtype=np.float64,
        )
        data = data[np.random.permutation(data.shape[0]), :]
        data = data[: data.shape[0] // 2]
        y_train = data[:, 0].astype(np.float64)
        X_train = data[:, 1:]
        return _finalize_train_only_dataset(X_train, y_train, dataset_name)

    if dataset_name == 'superconduct':
        X_train, y_train = fetch_openml(
            data_id=44148,
            as_frame=True,
            return_X_y=True,
            parser="auto",
        )
        y_train = y_train.astype(float).to_numpy(dtype=np.float64)
        X_train = X_train.to_numpy(dtype=np.float64)
        return _finalize_train_only_dataset(X_train, y_train, dataset_name)

    if dataset_name == 'airline':
        data = fetch_openml(data_id=42728, as_frame=True)
        df = data.frame.copy()
        df = df.sample(n=4194304, random_state=0)
        df = df.dropna()

        y_train = df["DepDelay"].astype(float).to_numpy(dtype=np.float64)
        X_train = df[["Distance", "DayOfWeek", "Month", "CRSDepTime"]].astype(float).to_numpy(dtype=np.float64)

        norm_fact_y = np.max(np.abs(y_train))
        if norm_fact_y > 0:
            y_train = y_train / norm_fact_y
        y_train *= 6.0
        y_train = np.clip(y_train, -1.0, 1.0)

        return _finalize_train_only_dataset(
            X_train,
            y_train,
            dataset_name,
            normalize_y=False,
        )

    if dataset_name == 'rossman':
        rossmann_df = pd.read_csv(
            os.path.join(dataset_path, 'rossman_train.csv'),
            low_memory=False,
        )
        store_df = pd.read_csv(
            os.path.join(dataset_path, 'rossman_store.csv'),
            low_memory=False,
        )

        rossmann_df['Date'] = pd.to_datetime(rossmann_df['Date'], errors='coerce')
        rossmann_df = rossmann_df.dropna(subset=['Date'])
        rossmann_df['Year'] = rossmann_df['Date'].dt.year.astype(np.int64)
        rossmann_df['Month'] = rossmann_df['Date'].dt.month.astype(np.int64)
        rossmann_df['Day'] = rossmann_df['Date'].dt.day.astype(np.int64)
        rossmann_df['WeekOfYear'] = rossmann_df['Date'].dt.isocalendar().week.astype(np.int64)

        rossmann_df['StateHoliday'] = rossmann_df['StateHoliday'].astype(str).replace({'0.0': '0'})
        store_df['StoreType'] = store_df['StoreType'].astype(str)
        store_df['Assortment'] = store_df['Assortment'].astype(str)
        if 'PromoInterval' in store_df.columns:
            store_df['PromoInterval'] = store_df['PromoInterval'].fillna('None').astype(str)

        numeric_store_cols = [
            'CompetitionDistance',
            'CompetitionOpenSinceMonth',
            'CompetitionOpenSinceYear',
            'Promo2SinceWeek',
            'Promo2SinceYear',
        ]
        for col in numeric_store_cols:
            if col in store_df.columns:
                store_df[col] = pd.to_numeric(store_df[col], errors='coerce')

        if 'CompetitionDistance' in store_df.columns:
            store_df['CompetitionDistance'] = store_df['CompetitionDistance'].fillna(
                store_df['CompetitionDistance'].median()
            )
        for col in ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear']:
            if col in store_df.columns:
                store_df[col] = store_df[col].fillna(0)

        df = pd.merge(rossmann_df, store_df, how='left', on='Store')
        if 'Open' in df.columns:
            df = df[df['Open'] != 0]
        df = df[df['Sales'] != 0]
        df = df.dropna(subset=['Sales'])

        def remove_outlier(df_in, col_name):
            q1 = df_in[col_name].quantile(0.25)
            q3 = df_in[col_name].quantile(0.75)
            iqr = q3 - q1
            fence_low = q1 - 1.5 * iqr
            fence_high = q3 + 1.5 * iqr
            return df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]

        df = remove_outlier(df, 'Sales').copy()

        drop_cols = ['Sales', 'Date']
        for col in ['Customers', 'Store', 'Year']:
            if col in df.columns:
                drop_cols.append(col)

        categorical_cols = [
            col for col in ['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval']
            if col in df.columns
        ]
        df_encoded = pd.get_dummies(
            df,
            columns=categorical_cols,
            drop_first=False,
            dtype=np.float64,
        )

        X_df = df_encoded.drop(columns=[c for c in drop_cols if c in df_encoded.columns]).copy()
        y_train = pd.to_numeric(df_encoded['Sales'], errors='coerce').to_numpy(dtype=np.float64)
        for col in X_df.columns:
            if X_df[col].dtype == bool:
                X_df[col] = X_df[col].astype(np.float64)
            elif not pd.api.types.is_numeric_dtype(X_df[col]):
                X_df[col] = pd.to_numeric(X_df[col], errors='coerce')

        X_df = X_df.fillna(X_df.median(numeric_only=True))
        X_df = X_df.fillna(0.0)
        X_train = X_df.to_numpy(dtype=np.float64)
        return _finalize_train_only_dataset(X_train, y_train, dataset_name)

    if dataset_name == 'black_friday':
        bunch = fetch_openml(data_id=41540, as_frame=True)
        df = bunch.frame.copy()

        y_train = df["Purchase"].to_numpy(dtype=np.float64)
        X_df = df.drop(columns=["Purchase"]).copy()
        X_df = X_df.drop(columns=["User_ID", "Product_ID"], errors="ignore")

        categorical_cols = [
            "Gender",
            "Age",
            "Occupation",
            "City_Category",
            "Stay_In_Current_City_Years",
            "Marital_Status",
            "Product_Category_1",
            "Product_Category_2",
            "Product_Category_3",
        ]
        existing_categorical_cols = [c for c in categorical_cols if c in X_df.columns]
        for col in existing_categorical_cols:
            X_df[col] = X_df[col].astype("string").fillna("missing")

        X_df = pd.get_dummies(X_df, columns=existing_categorical_cols, dummy_na=False)
        for col in X_df.columns:
            if X_df[col].dtype == bool:
                X_df[col] = X_df[col].astype(np.float64)
            elif not pd.api.types.is_numeric_dtype(X_df[col]):
                X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
        X_df = X_df.fillna(X_df.median(numeric_only=True))
        X_df = X_df.fillna(0.0)

        X_train = X_df.to_numpy(dtype=np.float64)
        return _finalize_train_only_dataset(X_train, y_train, dataset_name)

    if dataset_name in {'nyc-taxi-green-dec-2016', 'nyc_taxi_green_dec_2016'}:
        bunch = fetch_openml(data_id=42729, as_frame=True, parser="pandas")
        target_name = "tip_amount"

        bunch_target_name = getattr(bunch.target, "name", None)
        if isinstance(bunch_target_name, str) and bunch_target_name != "":
            target_name = bunch_target_name
        else:
            bunch_target_names = getattr(bunch, "target_names", None)
            if isinstance(bunch_target_names, str) and bunch_target_names != "":
                target_name = bunch_target_names
            elif isinstance(bunch_target_names, (list, tuple)) and len(bunch_target_names) > 0:
                target_name = bunch_target_names[0]

        X_df = bunch.data.copy() if bunch.data is not None else None
        if X_df is None and bunch.frame is not None:
            X_df = bunch.frame.drop(columns=[target_name], errors="ignore").copy()
        if X_df is None:
            raise ValueError("fetch_openml did not return feature data for nyc-taxi-green-dec-2016")

        existing_columns_to_drop = [
            col for col in ["total_amount", "lpep_pickup_datetime", "lpep_dropoff_datetime"]
            if col in X_df.columns
        ]
        if len(existing_columns_to_drop) > 0:
            X_df = X_df.drop(columns=existing_columns_to_drop)

        if bunch.target is not None:
            y_series = pd.to_numeric(pd.Series(bunch.target, name=target_name), errors="coerce")
        elif bunch.frame is not None and target_name in bunch.frame.columns:
            y_series = pd.to_numeric(bunch.frame[target_name], errors="coerce")
        else:
            raise ValueError(
                "Could not identify the target column returned by fetch_openml for nyc-taxi-green-dec-2016"
            )

        valid_target = y_series.notna()
        X_df = X_df.loc[valid_target].copy()
        y_train = y_series.loc[valid_target].to_numpy(dtype=np.float64)

        high_cardinality_categorical_cols = [
            col for col in ["PULocationID", "DOLocationID"] if col in X_df.columns
        ]
        low_cardinality_categorical_cols = [
            col
            for col in ["VendorID", "store_and_fwd_flag", "RatecodeID", "trip_type"]
            if col in X_df.columns
        ]
        categorical_cols = set(high_cardinality_categorical_cols + low_cardinality_categorical_cols)
        numeric_cols = [col for col in X_df.columns if col not in categorical_cols]

        X_processed = pd.DataFrame(index=X_df.index)
        for col in numeric_cols:
            X_processed[col] = pd.to_numeric(X_df[col].astype(str).str.strip(), errors="coerce")
        for col in categorical_cols:
            X_processed[col] = X_df[col].astype("string").str.strip().fillna("missing")

        continuous_parts = []
        if len(numeric_cols) > 0:
            X_numeric = X_processed[numeric_cols].astype(float)
            numeric_medians = X_numeric.median(axis=0, skipna=True).fillna(0.0)
            continuous_parts.append(X_numeric.fillna(numeric_medians))

        for col in high_cardinality_categorical_cols:
            values = X_processed[col].astype("string").fillna("missing")
            frequencies = values.value_counts(normalize=True)
            continuous_parts.append(
                pd.DataFrame(
                    {f"{col}_frequency": values.map(frequencies).fillna(0.0).astype(float)},
                    index=X_processed.index,
                )
            )

        X_parts = []
        if len(continuous_parts) > 0:
            X_continuous = pd.concat(continuous_parts, axis=1)
            continuous_means = X_continuous.mean(axis=0)
            continuous_scales = X_continuous.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
            X_continuous = (X_continuous - continuous_means) / continuous_scales
            X_parts.append(X_continuous.to_numpy(dtype=np.float64))

        for col in low_cardinality_categorical_cols:
            values = X_processed[col].astype("string").fillna("missing")
            categories = sorted(values.dropna().unique().tolist())
            for category in categories[1:]:
                X_parts.append((values == category).to_numpy(dtype=np.float64)[:, None])

        if len(X_parts) == 0:
            raise ValueError("NYC Taxi preprocessing produced no usable features")

        X_train = np.column_stack(X_parts)
        X_train = np.column_stack((X_train, np.ones(X_train.shape[0], dtype=np.float64)))
        return _finalize_train_only_dataset(X_train, y_train, dataset_name)

    if dataset_name == 'california':
        housing = pd.read_csv(
            os.path.join(dataset_path, "california.csv"),
            skipinitialspace=False,
        )
        housing.dropna(inplace=True)
        housing = housing.to_numpy()[:, :-1].astype(np.float64)
        X_train = housing[:, :-1]
        X_train = np.column_stack((X_train, np.ones(X_train.shape[0], dtype=np.float64)))
        y_train = housing[:, -1]
        return _finalize_train_only_dataset(X_train, y_train, dataset_name)

    if dataset_name == "beijing":
        beijing = pd.read_csv(
            os.path.join(dataset_path, "beijing.csv"),
            encoding="gb18030",
            usecols=[
                "DOM", "followers", "totalPrice", "square", "kitchen",
                "buildingType", "renovationCondition", "buildingStructure",
                "ladderRatio", "elevator", "fiveYearsProperty", "subway",
                "district", "communityAverage",
            ],
        )
        beijing = beijing.drop(index=60422, errors="ignore").copy()

        numeric_cols = [
            "DOM", "followers", "square", "kitchen",
            "ladderRatio", "communityAverage", "totalPrice",
        ]
        for col in numeric_cols:
            beijing[col] = pd.to_numeric(beijing[col], errors="coerce")

        for col in ["elevator", "fiveYearsProperty", "subway"]:
            beijing[col] = _to_numeric_binary(beijing[col])

        district_num = pd.to_numeric(beijing["district"], errors="coerce")
        if district_num.notna().all():
            beijing["district"] = district_num
        else:
            beijing["district"] = pd.factorize(
                beijing["district"].astype(str).str.strip()
            )[0].astype(float)

        beijing = beijing.dropna().copy()
        beijing = pd.get_dummies(
            beijing,
            columns=["buildingType", "renovationCondition", "buildingStructure"],
            drop_first=False,
            dtype=float,
        )

        y_train = beijing["totalPrice"].to_numpy(dtype=np.float64)
        X_train = beijing.drop(columns=["totalPrice"]).to_numpy(dtype=np.float64)
        X_train = np.column_stack((X_train, np.ones(X_train.shape[0], dtype=np.float64)))

        n_third = X_train.shape[0] // 3
        X_train = X_train[:n_third]
        y_train = y_train[:n_third]
        return _finalize_train_only_dataset(X_train, y_train, dataset_name)

    supported_datasets = [
        'years',
        'superconduct',
        'airline',
        'rossman',
        'black_friday',
        'nyc-taxi-green-dec-2016',
        'nyc_taxi_green_dec_2016',
        'california',
        'beijing',
    ]
    raise ValueError(
        f"Unknown dataset {dataset_name!r}. Supported real datasets: {supported_datasets}"
    )

################# Analytic Gaussian Mechanism #################
"""
    Taken from the official repository of Balle and Wang, ICML'18 
    https://github.com/BorjaBalle/analytic-gaussian-mechanism
"""

from math import exp, sqrt
from scipy.special import erf

def calibrateAnalyticGaussianMechanism(epsilon, delta, GS, tol = 1.e-12):
    """ Calibrate a Gaussian perturbation for differential privacy using the analytic Gaussian mechanism of [Balle and Wang, ICML'18]

    Arguments:
    epsilon : target epsilon (epsilon > 0)
    delta : target delta (0 < delta < 1)
    GS : upper bound on L2 global sensitivity (GS >= 0)
    tol : error tolerance for binary search (tol > 0)

    Output:
    sigma : standard deviation of Gaussian noise needed to achieve (epsilon,delta)-DP under global sensitivity GS
    """

    def Phi(t):
        return 0.5*(1.0 + erf(float(t)/sqrt(2.0)))

    def caseA(epsilon,s):
        return Phi(sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def caseB(epsilon,s):
        return Phi(-sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def doubling_trick(predicate_stop, s_inf, s_sup):
        while(not predicate_stop(s_sup)):
            s_inf = s_sup
            s_sup = 2.0*s_inf
        return s_inf, s_sup

    def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
        s_mid = s_inf + (s_sup-s_inf)/2.0
        while(not predicate_stop(s_mid)):
            if (predicate_left(s_mid)):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup-s_inf)/2.0
        return s_mid

    delta_thr = caseA(epsilon, 0.0)

    if (delta == delta_thr):
        alpha = 1.0

    else:
        if (delta > delta_thr):
            predicate_stop_DT = lambda s : caseA(epsilon, s) >= delta
            function_s_to_delta = lambda s : caseA(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) > delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) - sqrt(s/2.0)

        else:
            predicate_stop_DT = lambda s : caseB(epsilon, s) <= delta
            function_s_to_delta = lambda s : caseB(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) < delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) + sqrt(s/2.0)

        predicate_stop_BS = lambda s : abs(function_s_to_delta(s) - delta) <= tol

        s_inf, s_sup = doubling_trick(predicate_stop_DT, 0.0, 1.0)
        s_final = binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
        alpha = function_s_to_alpha(s_final)
        
    sigma = alpha*GS/sqrt(2.0*epsilon)

    return sigma

################################ Apply an ROS sketch and save the internal randomness ################################
class ROSMultiHDDraw:
    """
    One draw of a multi-layer ROS/SRFT-like sketch:

        F = sqrt(n/k) * P * (H D_T) (H D_{T-1}) ... (H D_1)

    where:
      - each D_t is an independent Rademacher diagonal (±1),
      - H is an orthonormal transform (DCT-II with norm='ortho' by default),
      - P selects k rows (idx).

    Shapes:
      X: (n, d)
      F X: (k, d)
      F^T Y: (n, d)
    """

    def __init__(self, n, k, T=1, transform="dct", norm="ortho", seed=None, rng=None):
        self.n = int(n)
        self.k = int(k)
        self.T = int(T)
        if self.T < 1:
            raise ValueError("T must be >= 1")
        self.transform = transform
        self.norm = norm

        if rng is not None and seed is not None:
            raise ValueError("Provide either seed or rng, not both.")
        if rng is None:
            rng = np.random.default_rng(seed)
        self.rng = rng

        self.rademachers = np.empty((self.T, self.n), dtype=np.int8)
        self.idx = np.empty(self.k, dtype=np.int64)

        self.scale = np.sqrt(self.n / self.k)
        self.scale_sq = self.n / self.k
        self.redraw()

    def redraw(self):
        self.rademachers[...] = self.rng.integers(
            0, 2, size=self.rademachers.shape, dtype=np.int8
        ) * 2 - 1
        self.idx[...] = self.rng.choice(self.n, size=self.k, replace=False)
        return self

    def _as_2d(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            return X[:, None]
        return X

    # ---------- H and H^T ----------
    def _apply_H(self, X):
        """Apply H to an (n, d) array."""
        if self.transform == "dct":
            return dct(X, type=2, n=self.n, axis=0, norm=self.norm)
        elif self.transform == "hadamard":
            return orthonormal_hadamard(X)
        else:
            raise ValueError("transform must be 'dct' or 'hadamard'")

    def _apply_Ht(self, X):
        """Apply H^T to an (n, d) array."""
        if self.transform == "dct":
            # For DCT-II with norm='ortho', the adjoint equals the inverse,
            # implemented via idct(type=2, norm='ortho') in SciPy.
            return idct(X, type=2, n=self.n, axis=0, norm=self.norm)
        elif self.transform == "hadamard":
            return orthonormal_hadamard(X)
        else:
            raise ValueError("transform must be 'dct' or 'hadamard'")

    # ---------- Core linear maps ----------
    def apply_A(self, X):
        """
        Apply A := (H D_T) ... (H D_1) to X (n,d) -> (n,d).
        """
        X = self._as_2d(X)
        if X.shape[0] != self.n:
            raise ValueError(f"X must have shape (n,d) with n={self.n}, got {X.shape}")

        if self.T == 1:
            return self._apply_H(X * self.rademachers[0][:, None])

        Z = X
        for signs in self.rademachers:
            Z = Z * signs[:, None]  # D_t
            Z = self._apply_H(Z)    # H
        return Z

    def apply_At(self, X):
        """
        Apply A^T to X (n,d) -> (n,d), where A=(H D_T)...(H D_1).
        Since (H D)^T = D H^T, we apply in reverse order:
            A^T = (D_1 H^T) ... (D_T H^T)
        """
        X = self._as_2d(X)
        if X.shape[0] != self.n:
            raise ValueError(f"X must have shape (n,d) with n={self.n}, got {X.shape}")

        if self.T == 1:
            return self._apply_Ht(X) * self.rademachers[0][:, None]

        Z = X
        for signs in self.rademachers[::-1]:
            Z = self._apply_Ht(Z)     # H^T
            Z = Z * signs[:, None]    # D_t
        return Z

    def apply_F(self, X):
        """
        Compute Y = F X = sqrt(n/k) * P * A X, shape (k, d).
        """
        AX = self.apply_A(X)                      # (n, d)
        Y = self.scale * AX[self.idx, :]          # (k, d)
        return Y

    def apply_Ft(self, Y):
        """
        Compute Z = F^T Y = sqrt(n/k) * A^T * P^T Y, shape (n, d).
        """
        Y = self._as_2d(Y)
        if Y.shape[0] != self.k:
            raise ValueError(f"Y must have shape (k,d) with k={self.k}, got {Y.shape}")

        # scatter: P^T Y
        Z0 = np.zeros((self.n, Y.shape[1]), dtype=Y.dtype)
        Z0[self.idx, :] = Y

        # apply A^T and scale
        Z = self.scale * self.apply_At(Z0)
        return Z

def _max_abs_offdiag_ftf_hadamard(Fdraw):
    z = np.zeros(Fdraw.n, dtype=np.float64)
    z[Fdraw.idx] = 1.0
    hz = orthonormal_hadamard(z)
    return float((np.sqrt(Fdraw.n) / Fdraw.k) * np.max(np.abs(hz[1:])))

def _max_abs_offdiag_ftf_dct(Fdraw):
    z = np.zeros(Fdraw.n, dtype=np.float64)
    z[Fdraw.idx] = 1.0

    dct_z = dct(z, type=2, n=Fdraw.n, norm=Fdraw.norm)
    cos_sums = np.empty(Fdraw.n, dtype=np.float64)
    cos_sums[0] = np.sqrt(Fdraw.n) * dct_z[0]
    if Fdraw.n > 1:
        cos_sums[1:] = np.sqrt(Fdraw.n / 2.0) * dct_z[1:]

    max_offdiag = 0.0
    if Fdraw.n > 1:
        max_offdiag = (np.sqrt(2.0) / Fdraw.k) * np.max(np.abs(cos_sums[1:]))
    if Fdraw.n <= 2:
        return float(max_offdiag)

    # For i,j > 0:
    #   (F^T F)_{ij} = (c_{|i-j|} + h_{i+j}) / k
    # where c_t = sum_{r in S} cos(pi(r+1/2)t/n) and
    # h_s = c_s for s < n, h_n = 0, h_s = -c_{2n-s} for s > n.
    hankel_values = np.empty(2 * Fdraw.n - 1, dtype=np.float64)
    hankel_values[:Fdraw.n] = cos_sums
    hankel_values[Fdraw.n] = 0.0
    if Fdraw.n > 2:
        hankel_values[Fdraw.n + 1:] = -cos_sums[Fdraw.n - 1:1:-1]

    max_difference = Fdraw.n - 2
    for parity in (0, 1):
        if max_difference < parity or (parity == 0 and max_difference < 2):
            continue

        current_diff = max_difference if (max_difference % 2 == parity) else max_difference - 1
        if current_diff < 1:
            continue

        parity_values = hankel_values[parity::2]
        left = (current_diff + 2 - parity) // 2
        right = (2 * Fdraw.n - 2 - current_diff - parity) // 2
        current_max = float(np.max(parity_values[left:right + 1]))
        current_min = float(np.min(parity_values[left:right + 1]))

        while current_diff >= 1:
            current_cos = cos_sums[current_diff]
            max_offdiag = max(
                max_offdiag,
                abs(current_cos + current_max) / Fdraw.k,
                abs(current_cos + current_min) / Fdraw.k,
            )

            current_diff -= 2
            if current_diff < 1:
                break

            left -= 1
            right += 1
            current_max = max(current_max, float(parity_values[left]), float(parity_values[right]))
            current_min = min(current_min, float(parity_values[left]), float(parity_values[right]))

    return float(max_offdiag)

def max_abs_offdiag_ftf(Fdraw):
    if Fdraw.T != 1:
        raise NotImplementedError("Exact max off-diagonal computation is implemented only for T=1.")

    if Fdraw.transform == "hadamard":
        return _max_abs_offdiag_ftf_hadamard(Fdraw)
    if Fdraw.transform == "dct":
        return _max_abs_offdiag_ftf_dct(Fdraw)
    raise ValueError("transform must be 'dct' or 'hadamard'")

def max_norm_Xt_FtF_minus_I_ei(X, Fdraw):
    """
    Compute the maximum rowwise quantity

        v_i = || X^T (F^T F - I_n) e_i ||_2,   i = 1,...,n

    and return that maximum together with the sketched matrix Y = F X.

    Since F = sqrt(n/k) P A with A orthonormal, we have

        (F^T F - I) X = A^T ( (n/k) P^T P - I ) A X.

    This routine computes that quantity exactly using the identity above,
    which avoids explicitly forming both F^T(FX) and F^T(FX) - X.

    Parameters
    ----------
    X : array-like, shape (n, d) or (n,)
        Data matrix.
    Fdraw : object
        Must support:
            - apply_F(X): returns F X
            - apply_Ft(Y): returns F^T Y
        and must have attribute:
            - n : number of rows of X

    Returns
    -------
    max_row_norm : float
        max_i || X^T (F^T F - I_n) e_i ||_2.
    max_abs_offdiag_ftf : float
        max_{i != j} | e_i^T F^T F e_j |.
    Y : ndarray, shape (k, d)
        The sketched matrix Y = F X.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]

    if X.shape[0] != Fdraw.n:
        raise ValueError(f"X must have n={Fdraw.n} rows, got {X.shape[0]}")

    # Exact rewrite:
    #   (F^T F - I)X = A^T(((n/k) P^T P - I) A X)
    U = Fdraw.apply_A(X)                                # shape (n, d)
    Y = U[Fdraw.idx, :].copy()                          # shape (k, d)
    Y *= Fdraw.scale                                    # shape (k, d)
    np.negative(U, out=U)                               # U <- -A X
    U[Fdraw.idx, :] += Fdraw.scale * Y
    diff = Fdraw.apply_At(U)                            # exact (F^T F - I)X

    # row i norm = || X^T(F^T F - I)e_i ||_2
    row_norm_sq = np.einsum("ij,ij->i", diff, diff, optimize=True)
    max_row_norm = float(np.sqrt(np.max(row_norm_sq)))
    max_offdiag_ftf = max_abs_offdiag_ftf(Fdraw)
    return max_row_norm, max_offdiag_ftf, Y

