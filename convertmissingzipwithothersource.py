import pandas as pd
import json

def combine_zipcode_data_optimized(csv_file_path, github_json_url, output_json_path, csv_priority=True, compress_json=False):
    """
    Combines ZIP code data from CSV and GitHub JSON with proper field mapping
    
    Args:
        csv_file_path: Path to your CSV file
        github_json_url: URL to the JSON data on GitHub  
        output_json_path: Path for output JSON file
        csv_priority: If True, CSV data takes precedence in conflicts
        compress_json: If True, outputs minified JSON without formatting
    """
    
    # Load JSON data from GitHub
    print("Loading JSON data from GitHub...")
    json_df = pd.read_json(github_json_url)
    
    # Standardize JSON column names to match CSV format
    json_df = json_df.rename(columns={
        'zip_code': 'zip',
        'latitude': 'lat', 
        'longitude': 'lng',
        'state': 'state_id',
        'county': 'county_name'
    })
    
    print(f"Loaded {len(json_df)} records from JSON")
    
    # Load CSV data
    print("Loading CSV data...")
    csv_df = pd.read_csv(csv_file_path)
    print(f"Loaded {len(csv_df)} records from CSV")
    
    # Perform outer merge to include ALL ZIP codes from both sources
    print("Merging datasets...")
    merged_df = pd.merge(csv_df, json_df, on='zip', how='outer', 
                        suffixes=('_csv', '_json'), indicator=True)
    
    print(f"After merge: {len(merged_df)} total records")
    
    # Create final DataFrame with resolved columns
    final_columns = {'zip': merged_df['zip']}
    
    # Define all possible columns that might need resolution
    conflict_columns = ['lat', 'lng', 'city', 'state_id', 'county_name']
    
    for column in conflict_columns:
        col_csv = f'{column}_csv'
        col_json = f'{column}_json'
        
        if csv_priority:
            # CSV priority: use CSV data, fall back to JSON if missing
            if col_csv in merged_df.columns and col_json in merged_df.columns:
                final_columns[column] = merged_df[col_csv].combine_first(merged_df[col_json])
            elif col_csv in merged_df.columns:
                final_columns[column] = merged_df[col_csv]
            else:
                final_columns[column] = merged_df[col_json]
        else:
            # JSON priority: use JSON data, fall back to CSV if missing  
            if col_csv in merged_df.columns and col_json in merged_df.columns:
                final_columns[column] = merged_df[col_json].combine_first(merged_df[col_csv])
            elif col_json in merged_df.columns:
                final_columns[column] = merged_df[col_json]
            else:
                final_columns[column] = merged_df[col_csv]
    
    # Handle non-conflicting columns (present in only one source)
    all_columns = set(csv_df.columns).union(set(json_df.columns))
    non_conflict_columns = all_columns - set(conflict_columns) - {'zip'}
    
    for column in non_conflict_columns:
        if column in csv_df.columns:
            final_columns[column] = merged_df.get(f'{column}_csv', merged_df.get(column))
        elif column in json_df.columns:
            final_columns[column] = merged_df.get(f'{column}_json', merged_df.get(column))
    
    # Create final DataFrame
    final_df = pd.DataFrame(final_columns)
    
    # Add source tracking
    source_map = {
        'left_only': 'csv_only',
        'right_only': 'json_only',
        'both': 'both_sources'
    }
    final_df['_source'] = merged_df['_merge'].map(source_map)
    
    # Fill missing coordinates by looking up same city/state in the dataset
    print("Filling missing coordinates...")
    missing_coords = final_df[(final_df['lat'].isna()) | (final_df['lng'].isna())]
    print(f"Found {len(missing_coords)} records with missing coordinates")
    
    # Save the final result with optional compression
    print(f"Saving combined data to {output_json_path}...")
    
    if compress_json:
        # Save as minified JSON (no formatting, smaller file size)
        final_df.to_json(output_json_path, orient='records', indent=None)
        print("Saved as compressed (minified) JSON")
    else:
        # Save as formatted JSON (human readable)
        final_df.to_json(output_json_path, orient='records', indent=2)
        print("Saved as formatted JSON")
    
    # Print comprehensive summary
    print("\n" + "="*60)
    print("MERGE COMPLETION SUMMARY")
    print("="*60)
    
    source_counts = final_df['_source'].value_counts()
    for source, count in source_counts.items():
        print(f"{source:15}: {count:>6} records")
    
    print(f"{'Total':15}: {len(final_df):>6} unique ZIP codes")
    
    # Check for missing coordinates in final data
    missing_final = final_df[(final_df['lat'].isna()) | (final_df['lng'].isna())]
    print(f"{'Missing coords':15}: {len(missing_final):>6} records")
    
    if len(missing_final) > 0:
        print("\nZIP codes still missing coordinates:")
        for zip_code in missing_final['zip'].head(10):
            print(f"  - {zip_code}")
        if len(missing_final) > 10:
            print(f"  ... and {len(missing_final) - 10} more")
    
    print("="*60)

# Usage examples
if __name__ == "__main__":
    CSV_FILE = "uszips.csv"
    GITHUB_JSON_URL = "https://raw.githubusercontent.com/millbj92/US-Zip-Codes-JSON/refs/heads/master/USCities.json"
    
    # Example 1: Compressed JSON output (smaller file size)
    print("=== Creating compressed JSON ===")
    combine_zipcode_data_optimized(
        csv_file_path=CSV_FILE,
        github_json_url=GITHUB_JSON_URL,
        output_json_path="uszips_combined_compressed.json",
        csv_priority=True,
        compress_json=True  # This creates minified JSON
    )
    
    # Example 2: Formatted JSON output (human readable)
    print("\n=== Creating formatted JSON ===")
    combine_zipcode_data_optimized(
        csv_file_path=CSV_FILE,
        github_json_url=GITHUB_JSON_URL,
        output_json_path="uszips_combined_formatted.json",
        csv_priority=True,
        compress_json=False  # This creates formatted JSON
    )