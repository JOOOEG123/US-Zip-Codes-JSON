import pandas as pd
import json
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from urllib.parse import urlparse
import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safe_json_loads(value, default=None):
    """Safely parse JSON strings with proper error handling"""
    if pd.isna(value) or value is None or value == '':
        return default
    
    try:
        # Handle already parsed JSON objects
        if isinstance(value, (dict, list)):
            return value
        
        # Clean and parse JSON string
        if isinstance(value, str):
            # Replace single quotes with double quotes for better compatibility
            cleaned_value = value.replace("'", '"').strip()
            return json.loads(cleaned_value)
        
        return default
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse JSON value: {value}. Error: {e}")
        return default

def download_json_from_url(url: str, timeout: int = 30) -> Optional[pd.DataFrame]:
    """Download and parse JSON from URL with proper error handling"""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        data = response.json()
        
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # Handle JSON objects with nested structure
            if 'data' in data and isinstance(data['data'], list):
                return pd.DataFrame(data['data'])
            elif 'features' in data and isinstance(data['features'], list):
                # Handle GeoJSON format
                features = data['features']
                records = []
                for feature in features:
                    if 'properties' in feature:
                        record = feature['properties'].copy()
                        if 'geometry' in feature and 'coordinates' in feature['geometry']:
                            coords = feature['geometry']['coordinates']
                            if coords and len(coords) >= 2:
                                record['lng'] = coords[0]
                                record['lat'] = coords[1]
                        records.append(record)
                return pd.DataFrame(records)
            else:
                # Flatten dictionary
                return pd.DataFrame([data])
        else:
            logger.error(f"Unexpected JSON structure: {type(data)}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download JSON from {url}: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response from {url}: {e}")
        return None

def optimize_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame data types to reduce memory usage"""
    if df.empty:
        return df
    
    df_optimized = df.copy()
    
    # Convert object columns to more efficient types
    for col in df_optimized.columns:
        if df_optimized[col].dtype == 'object':
            # Try to convert to numeric without using errors='ignore'
            try:
                # Attempt conversion - if it fails, it will raise an exception
                numeric_series = pd.to_numeric(df_optimized[col])
                df_optimized[col] = numeric_series
            except (ValueError, TypeError):
                # If conversion fails, check if we should convert to category
                if df_optimized[col].nunique() / len(df_optimized) < 0.5:
                    df_optimized[col] = df_optimized[col].astype('category')
    
    return df_optimized

def fill_missing_coordinates(final_df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing coordinates using various strategies"""
    if final_df.empty:
        return final_df
    
    df_filled = final_df.copy()
    
    # Ensure we have the required columns
    if 'lat' not in df_filled.columns or 'lng' not in df_filled.columns:
        logger.warning("Missing lat/lng columns, skipping coordinate filling")
        return df_filled
        
    if 'city' not in df_filled.columns or 'state_id' not in df_filled.columns:
        logger.warning("Missing city/state_id columns, skipping coordinate filling")
        return df_filled
    
    missing_before = ((df_filled['lat'].isna()) | (df_filled['lng'].isna())).sum()
    
    if missing_before == 0:
        return df_filled
    
    logger.info(f"Filling missing coordinates for {missing_before} records...")
    
    # Strategy 1: Use city/state averages
    # Ensure city and state_id are strings to avoid unhashable types
    df_filled['city'] = df_filled['city'].astype(str)
    df_filled['state_id'] = df_filled['state_id'].astype(str)
    
    city_state_coords = df_filled.groupby(['city', 'state_id']).agg({
        'lat': 'mean',
        'lng': 'mean'
    }).reset_index()
    
    city_state_coords = city_state_coords[
        (city_state_coords['lat'].notna()) & 
        (city_state_coords['lng'].notna())
    ]
    
    # Merge with original data to fill missing coordinates
    missing_mask = (df_filled['lat'].isna()) | (df_filled['lng'].isna())
    missing_data = df_filled[missing_mask].merge(
        city_state_coords, 
        on=['city', 'state_id'], 
        how='left',
        suffixes=('', '_fill')
    )
    
    # Fill coordinates
    fill_mask = missing_data['lat_fill'].notna()
    if fill_mask.any():
        df_filled.loc[missing_mask, 'lat'] = df_filled.loc[missing_mask, 'lat'].fillna(
            pd.Series(missing_data['lat_fill'].values, index=missing_data.index)
        )
        df_filled.loc[missing_mask, 'lng'] = df_filled.loc[missing_mask, 'lng'].fillna(
            pd.Series(missing_data['lng_fill'].values, index=missing_data.index)
        )
    
    missing_after = ((df_filled['lat'].isna()) | (df_filled['lng'].isna())).sum()
    logger.info(f"Filled {missing_before - missing_after} coordinate pairs")
    
    return df_filled

def combine_zipcode_data_optimized(
    csv_file_path: str, 
    github_json_url: str, 
    output_json_path: str, 
    csv_priority: bool = True, 
    compress_json: bool = False,
    batch_size: Optional[int] = None,
    fill_missing_coords: bool = True
) -> bool:
    """
    Combines ZIP code data from CSV and GitHub JSON with proper field mapping
    
    Args:
        csv_file_path: Path to your CSV file
        github_json_url: URL to the JSON data on GitHub  
        output_json_path: Path for output JSON file
        csv_priority: If True, CSV data takes precedence in conflicts
        compress_json: If True, outputs minified JSON without formatting
        batch_size: If specified, processes data in batches to reduce memory usage
        fill_missing_coords: Whether to fill missing coordinates
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    try:
        # Load JSON data from GitHub
        logger.info("Loading JSON data from GitHub...")
        json_df = download_json_from_url(github_json_url)
        
        if json_df is None or json_df.empty:
            logger.error("Failed to load JSON data or data is empty")
            return False
        
        # Standardize JSON column names to match CSV format
        column_mapping = {
            'zip_code': 'zip',
            'postal_code': 'zip',
            'latitude': 'lat', 
            'longitude': 'lng',
            'state': 'state_id',
            'state_abbr': 'state_id',
            'county': 'county_name',
            'county_fips': 'county_fips',
            'city_name': 'city',
            'place_name': 'city'
        }
        
        json_df = json_df.rename(columns={k: v for k, v in column_mapping.items() if k in json_df.columns})
        logger.info(f"Loaded {len(json_df)} records from JSON")
        
        # Load CSV data
        logger.info("Loading CSV data...")
        if batch_size:
            # Process CSV in chunks to reduce memory usage
            csv_chunks = []
            for chunk in pd.read_csv(csv_file_path, chunksize=batch_size):
                csv_chunks.append(optimize_data_types(chunk))
            csv_df = pd.concat(csv_chunks, ignore_index=True)
        else:
            csv_df = pd.read_csv(csv_file_path)
            csv_df = optimize_data_types(csv_df)
        
        logger.info(f"Loaded {len(csv_df)} records from CSV")
        
        # Ensure ZIP code is string type for proper merging
        for df in [csv_df, json_df]:
            if 'zip' in df.columns:
                df['zip'] = df['zip'].astype(str).str.zfill(5)  # Standardize to 5-digit format
        
        # Perform outer merge to include ALL ZIP codes from both sources
        logger.info("Merging datasets...")
        merged_df = pd.merge(csv_df, json_df, on='zip', how='outer', 
                            suffixes=('_csv', '_json'), indicator=True)
        
        logger.info(f"After merge: {len(merged_df)} total records")
        
        # Create final DataFrame with resolved columns
        final_columns = {'zip': merged_df['zip']}
        
        # Define all possible columns that might need resolution
        conflict_columns = ['lat', 'lng', 'city', 'state_id', 'county_name', 'county_fips']
        
        for column in conflict_columns:
            col_csv = f'{column}_csv'
            col_json = f'{column}_json'
            
            if csv_priority:
                # CSV priority: use CSV data, fall back to JSON if missing
                if col_csv in merged_df.columns and col_json in merged_df.columns:
                    final_columns[column] = merged_df[col_csv].combine_first(merged_df[col_json])
                elif col_csv in merged_df.columns:
                    final_columns[column] = merged_df[col_csv]
                elif col_json in merged_df.columns:
                    final_columns[column] = merged_df[col_json]
            else:
                # JSON priority: use JSON data, fall back to CSV if missing  
                if col_csv in merged_df.columns and col_json in merged_df.columns:
                    final_columns[column] = merged_df[col_json].combine_first(merged_df[col_csv])
                elif col_json in merged_df.columns:
                    final_columns[column] = merged_df[col_json]
                elif col_csv in merged_df.columns:
                    final_columns[column] = merged_df[col_csv]
        
        # Handle non-conflicting columns (present in only one source)
        all_columns = set(csv_df.columns).union(set(json_df.columns))
        non_conflict_columns = all_columns - set(conflict_columns) - {'zip'}
        
        for column in non_conflict_columns:
            col_csv = f'{column}_csv'
            col_json = f'{column}_json'
            
            if col_csv in merged_df.columns:
                final_columns[column] = merged_df[col_csv]
            elif col_json in merged_df.columns:
                final_columns[column] = merged_df[col_json]
            elif column in merged_df.columns:
                final_columns[column] = merged_df[column]
        
        # Create final DataFrame
        final_df = pd.DataFrame(final_columns)
        
        # Add source tracking
        source_map = {
            'left_only': 'csv_only',
            'right_only': 'json_only',
            'both': 'both_sources'
        }
        final_df['_source'] = merged_df['_merge'].map(source_map)
        
        # Process JSON fields in the final dataset - ensure they're strings for JSON serialization
        json_columns = [col for col in final_df.columns if 'json' in col.lower() or 'weights' in col.lower()]
        for col in json_columns:
            if col in final_df.columns:
                # Convert to string representation to avoid unhashable dict issues
                final_df[col] = final_df[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
                )
        
        # Convert potential dict/list columns to strings to avoid unhashable type errors
        for col in final_df.columns:
            # Sample a few rows to check for dict/list types
            sample_size = min(10, len(final_df))
            if sample_size > 0:
                sample = final_df[col].head(sample_size)
                has_dicts = any(isinstance(x, (dict, list)) for x in sample if pd.notna(x))
                if has_dicts:
                    logger.info(f"Converting {col} to string to avoid unhashable type issues")
                    final_df[col] = final_df[col].apply(
                        lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
                    )
        
        # Fill missing coordinates if requested
        if fill_missing_coords and 'lat' in final_df.columns and 'lng' in final_df.columns:
            final_df = fill_missing_coordinates(final_df)
        
        # Optimize final data types
        final_df = optimize_data_types(final_df)
        
        # Save the final result with optional compression
        logger.info(f"Saving combined data to {output_json_path}...")
        
        if compress_json:
            # Save as minified JSON (no formatting, smaller file size)
            final_df.to_json(output_json_path, orient='records', indent=None, index=False)
            logger.info("Saved as compressed (minified) JSON")
        else:
            # Save as formatted JSON (human readable)
            final_df.to_json(output_json_path, orient='records', indent=2, index=False)
            logger.info("Saved as formatted JSON")
        
        # Print comprehensive summary
        print_summary(final_df)
        return True
        
    except Exception as e:
        logger.error(f"Error during data combination: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def print_summary(final_df: pd.DataFrame):
    """Print comprehensive summary of the merge operation"""
    print("\n" + "="*60)
    print("MERGE COMPLETION SUMMARY")
    print("="*60)
    
    if '_source' in final_df.columns:
        source_counts = final_df['_source'].value_counts()
        for source, count in source_counts.items():
            print(f"{source:15}: {count:>6} records")
    
    print(f"{'Total':15}: {len(final_df):>6} unique ZIP codes")
    
    # Check for missing coordinates in final data
    if 'lat' in final_df.columns and 'lng' in final_df.columns:
        missing_final = final_df[(final_df['lat'].isna()) | (final_df['lng'].isna())]
        print(f"{'Missing coords':15}: {len(missing_final):>6} records")
        
        if len(missing_final) > 0:
            print("\nZIP codes still missing coordinates:")
            for zip_code in missing_final['zip'].head(10):
                print(f"  - {zip_code}")
            if len(missing_final) > 10:
                print(f"  ... and {len(missing_final) - 10} more")
    
    # Data quality metrics
    print(f"{'Duplicate ZIPs':15}: {(final_df['zip'].duplicated().sum()):>6} records")
    
    if 'state_id' in final_df.columns:
        state_count = final_df['state_id'].nunique()
        print(f"{'States covered':15}: {state_count:>6} states")
    
    print("="*60)

# Usage examples
if __name__ == "__main__":
    CSV_FILE = "uszips.csv"
    GITHUB_JSON_URL = "https://raw.githubusercontent.com/millbj92/US-Zip-Codes-JSON/refs/heads/master/USCities.json"
    
    # Example 1: Compressed JSON output with memory optimization
    print("=== Creating compressed JSON with batch processing ===")
    success = combine_zipcode_data_optimized(
        csv_file_path=CSV_FILE,
        github_json_url=GITHUB_JSON_URL,
        output_json_path="uszips_combined_compressed.json",
        csv_priority=True,
        compress_json=True,
        batch_size=5000,  # Process in batches of 5000 rows
        fill_missing_coords=True
    )
    
    if success:
        print("✓ Compression completed successfully")
    else:
        print("✗ Compression failed")
    
    # Example 2: Formatted JSON output with full processing
    print("\n=== Creating formatted JSON ===")
    success = combine_zipcode_data_optimized(
        csv_file_path=CSV_FILE,
        github_json_url=GITHUB_JSON_URL,
        output_json_path="uszips_combined_formatted.json",
        csv_priority=True,
        compress_json=False,
        fill_missing_coords=True
    )
    
    if success:
        print("✓ Formatting completed successfully")
    else:
        print("✗ Formatting failed")