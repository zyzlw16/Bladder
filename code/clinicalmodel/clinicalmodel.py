import pandas as pd  
import numpy as np  
from lifelines import CoxPHFitter  
from lifelines.utils import concordance_index  
from sklearn.utils import resample  
import warnings
warnings.filterwarnings("ignore")

warnings.simplefilter(action='ignore', category=FutureWarning)  

# Load data  
train_df = pd.read_csv(r"C:\Users")  
test_df = pd.read_csv(r"C:\Users")  
val_df = pd.read_csv(r"C:\Users")  
val_df2 = pd.read_csv(r"C:\Users") 

features = [''] 
duration_col = 'TIME'  
event_col = 'Status'  

cph = CoxPHFitter()  
cph.fit(train_df[[duration_col, event_col] + features], duration_col=duration_col, event_col=event_col)  

def calculate_c_index_ci(model, df, n_bootstraps=1000):  
    c_index_values = []  
    for _ in range(n_bootstraps):  
        # Resample with replacement  
        sample_df = resample(df)  
        # Calculate C-index  
        hazard_ratios = model.predict_partial_hazard(sample_df)  
        c_index_val = concordance_index(sample_df['TIME'], -hazard_ratios.values.ravel(), sample_df['Status'])  
        c_index_values.append(c_index_val)  

    lower_bound = np.percentile(c_index_values, 2.5)  
    upper_bound = np.percentile(c_index_values, 97.5)  
    return np.mean(c_index_values), lower_bound, upper_bound  

def evaluate_performance(model, df, dataset_name, pri=1, save_path=None):  
    hazard_ratios = model.predict_partial_hazard(df)  
    c_index, lower, upper = calculate_c_index_ci(model, df)  
    
    # Print C-index and its 95% CI  
    if pri:  
        print(f"C-index on {dataset_name}: {c_index:.4f} (95% CI: {lower:.4f}, {upper:.4f})")  
    
    # Save results to a CSV file, if save_path is provided  
    if save_path:  
        results_df = pd.DataFrame({  
            'Hazard Ratios': hazard_ratios.values.ravel(),  
        })  
        results_df.to_csv(save_path, index=False)  

# Evaluate and save each dataset's results  
evaluate_performance(cph, train_df, "", 1, r"C:\Users\.csv")  
evaluate_performance(cph, test_df, "", 1, r"C:\Users\.csv")  
evaluate_performance(cph, val_df, "", 1, r"C:\Users\.csv")  
evaluate_performance(cph, val_df2, "", 1, r"C:\Users\.csv")