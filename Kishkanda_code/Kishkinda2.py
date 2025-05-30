import pandas as pd

def convert_csv_to_txt(input_file, output_file):
    """
    Convert CSV file to formatted text file with optimized structure
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Clean column names (remove extra spaces)
    df.columns = df.columns.str.strip()
    
    # Drop duplicate rows based on all columns
    df = df.drop_duplicates()
    
    # Sort by Book, Chapter, and Verse for proper order
    
    
    # Open output file for writing
    with open(output_file, 'w', encoding='utf-8') as f:
        current_book = None
        current_chapter = None
        
        for _, row in df.iterrows():
            book = row['Kanda/Book']
            chapter = row['Sarga/Chapter'] 
            verse = row['Shloka/Verse Number']
            translation = row['English Translation']
            
            # Write book name only when it changes
            if book != current_book:
                if current_book is not None:  # Add spacing between books
                    f.write('\n\n')
                f.write(f'=== {book} ===\n\n')
                current_book = book
                current_chapter = None  # Reset chapter when book changes
            
            # Write sarga only when it changes
            if chapter != current_chapter:
                if current_chapter is not None:  # Add spacing between sargas
                    f.write('\n')
                f.write(f'SARGA {chapter}:\n')
                current_chapter = chapter
            
            # Write shlok number from original data and translation
            f.write(f'Shlok {verse}: {translation}\n')
    
    print(f"Successfully converted to {output_file}")
    print(f"Total verses processed: {len(df)}")

def convert_dataframe_to_txt(df, output_file):
    """
    Convert pandas DataFrame to formatted text file
    Use this if you already have the data in a DataFrame
    """
    # Clean column names (remove extra spaces)
    df.columns = df.columns.str.strip()
    
    # Drop duplicate rows
    df = df.drop_duplicates()
    
    # Sort by Book, Chapter, and Verse for proper order
    df = df.sort_values(['Kanda/Book', 'Sarga/Chapter', 'Shloka/Verse Number'])
    
    # Open output file for writing
    with open(output_file, 'w', encoding='utf-8') as f:
        current_book = None
        current_chapter = None
        
        for _, row in df.iterrows():
            book = row['Kanda/Book']
            chapter = row['Sarga/Chapter'] 
            verse = row['Shloka/Verse Number']
            translation = row['English Translation']
            
            # Write book name only when it changes
            if book != current_book:
                if current_book is not None:  # Add spacing between books
                    f.write('\n\n')
                f.write(f'=== {book} ===\n\n')
                current_book = book
                current_chapter = None  # Reset chapter when book changes
            
            # Write chapter only when it changes
            if chapter != current_chapter:
                if current_chapter is not None:  # Add spacing between chapters
                    f.write('\n')
                f.write(f'Chapter {chapter}:\n')
                current_chapter = chapter
            
            # Write verse number and translation
            f.write(f'{verse}. {translation}\n')
    
    print(f"Successfully converted to {output_file}")
    print(f"Total verses processed: {len(df)}")

# Example usage:
if __name__ == "__main__":
    # Method 1: If you have a CSV file
    convert_csv_to_txt('C:/PF/Projects/NYD/Datasets/Kishkinda_kanda_final.csv', 'output.txt')
    
    # Method 2: If you have a DataFrame (example with sample data)
    sample_data = {
        'Kanda/Book': ['KISHKINDA KANI', 'KISHKINDA KANI', 'KISHKINDA KANI', 'KISHKINDA KANI'],
        'Sarga/Chapter': [1, 1, 1, 2],
        'Shloka/Verse Number': [1, 2, 3, 1],
        'English Translation': [
            'Well mingled are these female birds with their male ones...',
            'Rama, on arriving at that Lake of Lotuses called Pampa...',
            'There, on seeing that Lake of Lotuses, thrilled are his senses...',
            'Oh! Soumitri, magnificent is Pampa Lake...'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    convert_dataframe_to_txt(df, 'sample_output.txt')
    
    print("\nTo use with your actual data:")
    print("1. Save your CSV file and use: convert_csv_to_txt('your_file.csv', 'output.txt')")
    print("2. Or if you have a DataFrame: convert_dataframe_to_txt(your_df, 'output.txt')")