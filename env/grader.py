import pandas as pd
import numpy as np

def compare_columns(output_df: pd.DataFrame, expected_df: pd.DataFrame) -> float:
    try:
        expected_cols = set(expected_df.columns)
        output_cols = set(output_df.columns)
        if not expected_cols:
            return 0.0
        common = expected_cols.intersection(output_cols)
        return len(common) / len(expected_cols)
    except:
        return 0.0

def compare_values(output_df: pd.DataFrame, expected_df: pd.DataFrame) -> float:
    try:
        out_aligned = output_df.reset_index(drop=True)
        exp_aligned = expected_df.reset_index(drop=True)
        if len(exp_aligned) == 0:
            return 1.0 if len(out_aligned) == 0 else 0.0
        
        common_cols = list(set(expected_df.columns).intersection(output_df.columns))
        if not common_cols:
            return 0.0
            
        min_len = min(len(out_aligned), len(exp_aligned))
        if min_len == 0:
            return 0.0
            
        out_subset = out_aligned.loc[:min_len-1, common_cols]
        exp_subset = exp_aligned.loc[:min_len-1, common_cols]
        
        matches = (out_subset == exp_subset).sum().sum()
        total_cells = len(exp_aligned) * len(expected_df.columns)
        num_matches = matches + (out_subset.isna() & exp_subset.isna()).sum().sum()
        
        # Penalize for row count mismatch safely
        penalty = max(0, abs(len(out_aligned) - len(exp_aligned)))
        
        score = float((num_matches) / total_cells) - (penalty * 0.1)
        return max(score, 0.0)
    except Exception:
        return 0.0

def compare_types(output_df: pd.DataFrame, expected_df: pd.DataFrame) -> float:
    try:
        expected_cols = set(expected_df.columns)
        if not expected_cols:
            return 0.0
        
        matches = 0
        for col in expected_cols:
            if col in output_df.columns:
                if output_df[col].dtype == expected_df[col].dtype:
                    matches += 1
        return matches / len(expected_cols)
    except:
        return 0.0

def grade(output_df: pd.DataFrame, expected_df: pd.DataFrame) -> float:
    column_score = compare_columns(output_df, expected_df)
    value_score = compare_values(output_df, expected_df)
    type_score = compare_types(output_df, expected_df)

    total_score = (
        0.3 * column_score +
        0.5 * value_score +
        0.2 * type_score
    )

    return min(max(total_score, 0.0), 1.0)
