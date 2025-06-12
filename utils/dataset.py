
import os
import pickle
from pathlib import Path


def update_shapely_pickles(base_dir):
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == "forest_loss_region.pkl":
                pkl_path = Path(root) / file
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                with open(pkl_path, "wb") as f:
                    pickle.dump(data, f)


def filter_timeseries_by_length(df, min_length=2):
    """
    Filters time series data to only include locations with a specified number of steps.

    Args:
        df: Dataframe to process
        min_length: Minimum time series length to keep (default to 2 which will be all timeseries if not provided)

    Returns:
        Filtered DataFrame and statistics
    """
    # Extract location IDs (without timestep)
    df['location_id'] = df['example_path'].str.extract(r'(.+)_\d+$')

    # Get value counts and filter
    location_counts = df['location_id'].value_counts()
    valid_locations = location_counts[location_counts >= min_length].index

    # Filter original DataFrame
    filtered_df = df[df['location_id'].isin(valid_locations)].copy()

    # Calculate statistics
    stats = {
        'total_locations': len(location_counts),
        'filtered_locations': len(valid_locations),
        'min_length': min_length,
        'filtered_percentage': len(valid_locations) / len(location_counts)
    }

    return filtered_df, stats
