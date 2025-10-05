import csv
import json
import sys

def csv_to_json_large(csv_file_path, json_file_path):
    """
    Convert a large CSV file to JSON efficiently using streaming
    """
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
            # Create CSV reader
            csv_reader = csv.DictReader(csv_file)
            
            # Open JSON file for writing
            with open(json_file_path, 'w', encoding='utf-8') as json_file:
                # Start JSON array
                json_file.write('[\n')
                
                first_row = True
                
                # Process rows one by one to minimize memory usage
                for row in csv_reader:
                    if not first_row:
                        json_file.write(',\n')
                    else:
                        first_row = False
                    
                    # Convert specific numeric fields
                    if row['population']:
                        row['population'] = int(row['population'])
                    if row['density']:
                        row['density'] = float(row['density'])
                    if row['lat']:
                        row['lat'] = float(row['lat'])
                    if row['lng']:
                        row['lng'] = float(row['lng'])
                    
                    # Parse the county_weights JSON string
                    if row['county_weights']:
                        row['county_weights'] = json.loads(row['county_weights'].replace("'", '"'))
                    
                    # Convert boolean fields
                    row['zcta'] = row['zcta'].lower() == 'true'
                    row['imprecise'] = row['imprecise'].lower() == 'true'
                    row['military'] = row['military'].lower() == 'true'
                    
                    # Write individual row as JSON
                    json.dump(row, json_file, ensure_ascii=False, separators=(',', ':'))
                
                # End JSON array
                json_file.write('\n]')
                
        print(f"Successfully converted {csv_file_path} to {json_file_path}")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        sys.exit(1)

def csv_to_json_memory_optimized(csv_file_path, json_file_path, batch_size=1000):
    """
    Alternative version that processes in batches for very large files
    """
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            
            with open(json_file_path, 'w', encoding='utf-8') as json_file:
                json_file.write('[\n')
                
                batch = []
                first_batch = True
                
                for row in csv_reader:
                    # Process row data types
                    if row['population']:
                        row['population'] = int(row['population'])
                    if row['density']:
                        row['density'] = float(row['density'])
                    if row['lat']:
                        row['lat'] = float(row['lat'])
                    if row['lng']:
                        row['lng'] = float(row['lng'])
                    
                    if row['county_weights']:
                        row['county_weights'] = json.loads(row['county_weights'].replace("'", '"'))
                    
                    row['zcta'] = row['zcta'].lower() == 'true'
                    row['imprecise'] = row['imprecise'].lower() == 'true'
                    row['military'] = row['military'].lower() == 'true'
                    
                    batch.append(row)
                    
                    # Write in batches to manage memory
                    if len(batch) >= batch_size:
                        if not first_batch:
                            json_file.write(',\n')
                        else:
                            first_batch = False
                        
                        json_str = ',\n'.join(json.dumps(item, ensure_ascii=False, separators=(',', ':')) for item in batch)
                        json_file.write(json_str)
                        batch = []
                
                # Write remaining rows
                if batch:
                    if not first_batch:
                        json_file.write(',\n')
                    json_str = ',\n'.join(json.dumps(item, ensure_ascii=False, separators=(',', ':')) for item in batch)
                    json_file.write(json_str)
                
                json_file.write('\n]')
                
        print(f"Successfully converted {csv_file_path} to {json_file_path}")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        sys.exit(1)

# Usage example
if __name__ == "__main__":
    csv_file = "uszips.csv"
    json_file = "uszips.json"
    
    # Use the streaming version for most cases
    # csv_to_json_large(csv_file, json_file)
    
    # For extremely large files, use the batched version
    csv_to_json_memory_optimized(csv_file, json_file, batch_size=1000)