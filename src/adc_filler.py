import numpy as np
import pandas as pd

def complete_adc_df(df: pd.DataFrame, t0 = None) -> pd.DataFrame:
    """
    Takes an ADC DataFrame, finds timestamp gaps and returns a new DataFrame with
    the missing rows filled with NaNs.

    Args:
        df     (pd.DataFrame): Input ADC DataFrame with missing timestamps.

    Returns:
        new_df (pd.DataFrame): Output ADC DataFrame with missed timestamp rows filled with NaNs.
    """
    new_df = df.copy()
    #set time index to be relative to the first one
    if t0 is None:
        new_df.index = new_df.index - new_df.index[0]
    else:
        new_df.index = new_df.index - t0

    #Extract the columns and index as np.arrays for convenience
    columns = new_df.columns.to_numpy()
    index = new_df.index.to_numpy()

    #Get min and max timestamp in range
    min_index = index[0]
    max_index = index[-1]

    #Get complete list of timestamps there should be
    all_index = np.arange(min_index, max_index, 32, dtype=int)
    missing_index = np.setdiff1d(all_index, index)

    #Create NaNs DF for all missing rows
    gap_df = pd.DataFrame(np.nan, columns=columns, index=missing_index)

    #Merge with ADC DF to create new DF with gaps
    new_df = pd.concat([new_df, gap_df]).sort_index()
    new_df.index = new_df.index.astype(int)

    return new_df

def interpolate_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes an ADC DataFrame with NaNs in some rows, looks for them and perform a
    linear interpolation to complete these entries if the gap is less than 4 rows long.

    Args:
        df     (pd.DataFrame): Input ADC DataFrame with NaN entries.

    Returns:
        new_df (pd.DataFrame): Output ADC DataFrame with gaps filled by interpolation.
    """
    new_df = df.copy()
    empty_idx = np.flatnonzero(new_df.isna().any(axis=1))
    for idx in empty_idx:
        for i in range(1,5):
            if (idx+i not in empty_idx) and (idx+i < new_df.shape[0]):
                y1 = new_df.iloc[idx-1]
                y2 = new_df.iloc[idx+i]
                for j in range(i):
                    new_df.iloc[idx+j] = ((y2*(j+1)+((i+1)-(j+1))*y1)/(i+1)).astype(int)
                    empty_idx = empty_idx[empty_idx!=idx+j]
                break
            elif (idx+i not in empty_idx) and (idx+i >= new_df.shape[0]):
                new_df.iloc[idx] = new_df.iloc[idx-1]
                empty_idx = empty_idx[empty_idx!=idx]
            elif (idx+i not in empty_idx) and (i == 3):
                raise
    return new_df