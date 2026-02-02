
import csv
import os

INPUT_FILE = 'data/pittsburgh+bridges/bridges.data.version2'           
OUTPUT_FILE = 'data/pittsburgh+bridges/bridges.csv'       
COLUMN_NAMES = ["IDENTIF", "RIVER", "LOCATION", "ERECTED", "PURPOSE", "LENGTH",
                "LANES", "CLEAR-G", "T-OR-D", "MATERIAL", "SPAN", "REL-L", "TYPE"]  
# columns relating to the dataset being converted:



def convert_data_to_csv(input_file, output_file, column_names):

    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
        
        if not lines:
            print(f"Warning: {input_file} is empty")
            return
        
        # Parse the data
        data_rows = []
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                row = [value.strip() for value in line.split(',')]
                data_rows.append(row)
        
        if not data_rows:
            print(f"Warning: No data found in {input_file}")
            return
        
        # Check if column count matches
        first_row_length = len(data_rows[0])
        if len(column_names) != first_row_length:
            print(f"Warning: Number of columns ({len(column_names)}) doesn't match data columns ({first_row_length})")
            print("Adjusting column names...")
            
            if len(column_names) < first_row_length:
                # Add generic column names
                for i in range(len(column_names), first_row_length):
                    column_names.append(f"Column_{i+1}")
            else:
                # Truncate column names
                column_names = column_names[:first_row_length]
        
        # Write to CSV file
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            
            # Write header
            writer.writerow(column_names)
            
            # Write data rows
            writer.writerows(data_rows)
        
        print(f"Successfully converted {input_file} to {output_file}")
        print(f"Total rows: {len(data_rows)} (excluding header)")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
    except Exception as e:
        print(f"Error: {e}")


def main():
    
    print(f"Input file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Column names: {', '.join(COLUMN_NAMES)}")
    print()
    
    convert_data_to_csv(INPUT_FILE, OUTPUT_FILE, COLUMN_NAMES)


if __name__ == "__main__":
    main()