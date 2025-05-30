import re
import pandas as pd
import os
import glob

def parse_entry_block(block_text, current_book=None, current_sarga=None):
    """
    Parses a single block of text representing one verse entry.
    Extracts Kanda/Book, Sarga/Chapter, Shloka/Verse Number, and English Translation.
    Enhanced to handle Ramayana-specific formats like "SARGA X" and "Shlok X:"
    """
    entry = {
        'Kanda/Book': current_book,
        'Sarga/Chapter': current_sarga,
        'Shloka/Verse Number': None,
        'English Translation': None,
        'Order': None  # Add order field for chronological sorting
    }
    
    lines = block_text.strip().split('\n')
    translation_lines = []
    parsing_translation = False

    for line in lines:
        line = line.strip()
        if not line: # Skip empty lines that might be part of the block internally
            continue

        # Check for Kanda/Book (original format)
        kanda_match = re.match(r"^(?:Kanda/Book|KANDA)\s*:\s*(.*)", line, re.IGNORECASE)
        if kanda_match:
            entry['Kanda/Book'] = kanda_match.group(1).strip()
            parsing_translation = False
            continue

        # Check for Sarga/Chapter (original format)
        sarga_match = re.match(r"^(?:Sarga/Chapter|SARGA)\s*:\s*(.*)", line, re.IGNORECASE)
        if sarga_match:
            entry['Sarga/Chapter'] = sarga_match.group(1).strip()
            parsing_translation = False
            continue

        # Check for Shloka/Verse Number (original format)
        shloka_match = re.match(r"^(?:Shloka/Verse Number|SHLOKA)\s*:\s*(.*)", line, re.IGNORECASE)
        if shloka_match:
            entry['Shloka/Verse Number'] = shloka_match.group(1).strip()
            parsing_translation = False
            continue

        # NEW: Enhanced Ramayana-style Shlok format with improved regex
        # Handles all whitespace permutations: "Shlok 1:", "Shlok 18b-19a:", "Shlok 18b - 19a:", "Shlok 24b- 25- 26:", etc.
        ramayana_shlok_match = re.match(r'^Shlok\s+(\d+[a-z]?(?:\s*-\s*\d+[a-z]?)*)\s*:\s*(.*)', line)
        if ramayana_shlok_match:
            verse_num = ramayana_shlok_match.group(1)
            # Clean up whitespaces in verse number (normalize spaces around hyphens)
            verse_num = re.sub(r'\s*-\s*', '-', verse_num)
            entry['Shloka/Verse Number'] = verse_num
            translation_text = ramayana_shlok_match.group(2).strip()
            if translation_text:
                translation_lines = [translation_text]
            parsing_translation = True
            continue
            
        # Check for English Translation header (original format)
        translation_header_match = re.match(r"^(?:English Translation|TRANSLATION)\s*:(.*)", line, re.IGNORECASE)
        if translation_header_match:
            parsing_translation = True
            initial_translation_part = translation_header_match.group(1).strip()
            if initial_translation_part:
                translation_lines.append(initial_translation_part)
            continue 

        # If we are in parsing_translation mode, append the line
        if parsing_translation:
            # Check if this line starts a new field (using improved pattern)
            if re.match(r"^(?:Kanda/Book|KANDA|Sarga/Chapter|SARGA|Shloka/Verse Number|SHLOKA|Shlok\s+\d+[a-z]?(?:\s*-\s*\d+[a-z]?)*)\s*:", line, re.IGNORECASE):
                parsing_translation = False 
            else:
                translation_lines.append(line)
            
    if translation_lines:
        # Clean up the translation text
        entry['English Translation'] = " ".join(translation_lines).strip()
        entry['English Translation'] = re.sub(r'\s+', ' ', entry['English Translation'])
        
    # Only return a valid entry if at least the verse number and translation are present
    if entry['Shloka/Verse Number'] and entry['English Translation']:
        return entry
    return None

def parse_verse_number_for_sorting(verse_num):
    """
    Parse verse number into components for proper chronological sorting.
    Handles formats like: "1", "1a", "1b", "1-2", "1a-2b", "24b-25-26", etc.
    Returns a tuple that can be used for sorting chronologically.
    """
    if not verse_num:
        return (float('inf'),)
    
    # Split by hyphens to handle ranges
    parts = verse_num.split('-')
    sort_key = []
    
    for part in parts:
        part = part.strip()
        # Extract number and optional letter
        match = re.match(r'(\d+)([a-z]?)', part)
        if match:
            num = int(match.group(1))
            letter = match.group(2) if match.group(2) else ''
            # Convert letter to numeric value ('' = 0, 'a' = 1, 'b' = 2, etc.)
            letter_val = ord(letter) - ord('a') + 1 if letter else 0
            sort_key.append((num, letter_val))
        else:
            # Fallback for unexpected formats
            sort_key.append((float('inf'), 0))
    
    return tuple(sort_key)

def parse_text_file_to_data(file_content, default_book="Unknown Kanda"):
    """
    Enhanced parser that handles both original format and Ramayana-specific format.
    Can parse SARGA headers and maintain context across verses.
    Maintains chronological order as verses appear in the text.
    """
    parsed_data = []
    current_book = default_book
    current_sarga = None
    
    # First, try to detect if this is Ramayana format by looking for SARGA headers
    has_sarga_format = bool(re.search(r'^SARGA\s+\d+', file_content, re.MULTILINE))
    
    if has_sarga_format:
        # Parse Ramayana format with SARGA sections
        return parse_ramayana_format(file_content, default_book)
    else:
        # Use original block-based parsing
        blocks = re.split(r'\n\s*\n+', file_content.strip())
        
        order_counter = 0
        for block in blocks:
            if block.strip():
                entry_data = parse_entry_block(block, current_book, current_sarga)
                if entry_data:
                    # Update current context if new values are found
                    if entry_data['Kanda/Book']:
                        current_book = entry_data['Kanda/Book']
                    if entry_data['Sarga/Chapter']:
                        current_sarga = entry_data['Sarga/Chapter']
                    
                    # Assign chronological order
                    entry_data['Order'] = order_counter
                    order_counter += 1
                    
                    parsed_data.append(entry_data)
        
        return parsed_data

def parse_ramayana_format(file_content, default_book):
    """
    Specialized parser for Ramayana text format with SARGA headers and Shlok entries.
    Uses improved regex patterns to handle all whitespace permutations in verse numbers.
    Maintains chronological order as verses appear in the text.
    """
    parsed_data = []
    current_book = default_book
    current_sarga = None
    
    # Initialize regex patterns
    book_pattern = re.compile(r'(BALA|AYODHYA|ARANYA|KISHKINDA|SUNDARA|YUDDHA)\s+KANDA')
    sarga_pattern = re.compile(r'SARGA\s+(\d+)')
    
    lines = file_content.split('\n')
    i = 0
    order_counter = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines and separators
        if not line or line.startswith('---'):
            i += 1
            continue
        
        # Check for Book/Kanda header
        book_match = book_pattern.search(line)
        if book_match:
            current_book = book_match.group(0)
            i += 1
            continue
        
        # Check for SARGA header
        sarga_match = sarga_pattern.search(line)
        if sarga_match:
            current_sarga = int(sarga_match.group(1))
            i += 1
            continue
        
        # Check for Shlok entry with improved pattern that handles all whitespace permutations
        shlok_match = re.match(r'^Shlok\s+(\d+[a-z]?(?:\s*-\s*\d+[a-z]?)*)\s*:\s*(.*)', line)
        if shlok_match and current_sarga is not None:
            verse_num = shlok_match.group(1)
            # Clean up whitespaces around hyphens in verse number
            verse_num = re.sub(r'\s*-\s*', '-', verse_num)
            translation = shlok_match.group(2).strip()
            
            # Continue reading multi-line translation
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                # Stop if we hit a new Shlok, SARGA, or commentary (using improved pattern)
                if (not next_line or 
                    re.match(r'^(SARGA|Shlok\s+\d+[a-z]?(?:\s*-\s*\d+[a-z]?)*\s*:|\[Text\]|\[Commentary\])', next_line) or
                    next_line.startswith('---') or
                    book_pattern.search(next_line)):
                    break
                translation += " " + next_line
                i += 1
            
            # Clean up translation
            translation = re.sub(r'\s+', ' ', translation).strip()
            
            # Add entry with chronological order
            if translation:  # Only add if we have actual translation text
                entry = {
                    'Kanda/Book': current_book,
                    'Sarga/Chapter': current_sarga,
                    'Shloka/Verse Number': verse_num,
                    'English Translation': translation,
                    'Order': order_counter
                }
                parsed_data.append(entry)
                order_counter += 1
            continue
        
        i += 1
    
    return parsed_data

def process_multiple_files(file_paths):
    """
    Process multiple text files and combine their data.
    Returns a list of all parsed entries from all files.
    Maintains chronological order across files.
    """
    all_data = []
    processed_files = []
    global_order_counter = 0
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File '{file_path}' not found. Skipping...")
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            # Auto-detect book name from filename
            detected_book = detect_book_name_from_filename(file_path)
            
            print(f"Processing: {file_path} (Book: {detected_book})")
            
            # Parse the file
            file_data = parse_text_file_to_data(file_content, detected_book)
            
            if file_data:
                # Update global order for each entry
                for entry in file_data:
                    entry['Global_Order'] = global_order_counter
                    global_order_counter += 1
                
                all_data.extend(file_data)
                processed_files.append((file_path, len(file_data)))
                print(f"  → Parsed {len(file_data)} verses")
            else:
                print(f"  → No data found in {file_path}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    print(f"\nProcessing Summary:")
    print(f"Files processed: {len(processed_files)}")
    for file_path, count in processed_files:
        print(f"  - {os.path.basename(file_path)}: {count} verses")
    print(f"Total verses: {len(all_data)}")
    
    return all_data

def get_file_paths():
    """
    Get file paths from user input. Supports multiple input methods.
    """
    print("Choose input method:")
    print("1. Enter individual file paths")
    print("2. Enter directory path (processes all .txt files)")
    print("3. Enter file pattern (e.g., *.txt or ramayana_*.txt)")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        # Individual file paths
        file_paths = []
        print("Enter file paths one by one (press Enter with empty line to finish):")
        while True:
            path = input("File path: ").strip()
            if not path:
                break
            file_paths.append(path)
        return file_paths
        
    elif choice == "2":
        # Directory path
        dir_path = input("Enter directory path: ").strip()
        if os.path.isdir(dir_path):
            file_paths = glob.glob(os.path.join(dir_path, "*.txt"))
            if file_paths:
                print(f"Found {len(file_paths)} .txt files in directory:")
                for path in file_paths:
                    print(f"  - {os.path.basename(path)}")
                return file_paths
            else:
                print("No .txt files found in directory.")
                return []
        else:
            print("Invalid directory path.")
            return []
            
    elif choice == "3":
        # File pattern
        pattern = input("Enter file pattern (e.g., *.txt, ramayana_*.txt): ").strip()
        file_paths = glob.glob(pattern)
        if file_paths:
            print(f"Found {len(file_paths)} files matching pattern:")
            for path in file_paths:
                print(f"  - {os.path.basename(path)}")
            return file_paths
        else:
            print("No files found matching pattern.")
            return []
    else:
        print("Invalid choice.")
        return []

def detect_book_name_from_filename(file_path):
    """
    Attempts to detect the book name from the filename.
    """
    book_mappings = {
        'bala': 'Bala Kanda',
        'ayodhya': 'Ayodhya Kanda', 
        'aranya': 'Aranya Kanda',
        'kishkindha': 'Kishkindha Kanda',
        'sundara': 'Sundara Kanda',
        'lanka': 'Lanka Kanda',
        'yuddha': 'Yuddha Kanda',
        'uttara': 'Uttara Kanda'
    }
    
    filename_lower = file_path.lower()
    for key, book_name in book_mappings.items():
        if key in filename_lower:
            return book_name
    
    return "Unknown Kanda"

def main():
    """
    Enhanced main function to handle multiple files with better organization.
    Maintains chronological order of shlokas as they appear in the text.
    """
    print("=== Ramayana Text Parser - Multiple Files Support (Chronological Order) ===\n")
    
    # Get file paths
    file_paths = get_file_paths()
    
    if not file_paths:
        print("No files to process. Exiting.")
        return
    
    # Get output file path
    default_output = "ramayana_combined.xlsx"
    output_file_path = input(f"Enter output Excel file path (default: {default_output}): ").strip()
    if not output_file_path:
        output_file_path = default_output
    
    # Ensure output file has .xlsx extension
    if not output_file_path.endswith('.xlsx'):
        output_file_path += '.xlsx'
    
    print(f"\n=== Processing Files ===")
    
    # Process all files
    all_data = process_multiple_files(file_paths)
    
    if not all_data:
        print("No data parsed from any files. Please check file formats and content.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Ensure columns are in desired order
    columns_ordered = ['Kanda/Book', 'Sarga/Chapter', 'Shloka/Verse Number', 'English Translation']
    df_columns_present = [col for col in columns_ordered if col in df.columns]
    df = df[df_columns_present]
    
    # Sort the data chronologically (maintaining text order)
    if 'Global_Order' in df.columns:
        # Use global order to maintain chronological sequence across files
        df = df.sort_values('Global_Order').drop('Global_Order', axis=1)
    elif 'Order' in df.columns:
        # Use local order within files
        df = df.sort_values('Order').drop('Order', axis=1)
    else:
        # Fallback: sort by book, chapter, and chronological verse order
        if 'Kanda/Book' in df.columns and 'Sarga/Chapter' in df.columns:
            # Create a custom sort order for Kandas
            kanda_order = {
                'Bala Kanda': 1, 'Ayodhya Kanda': 2, 'Aranya Kanda': 3,
                'Kishkindha Kanda': 4, 'Sundara Kanda': 5, 'Lanka Kanda': 6,
                'Yuddha Kanda': 6, 'Uttara Kanda': 7, 'Unknown Kanda': 8
            }
            df['sort_order'] = df['Kanda/Book'].map(kanda_order).fillna(9)
            
            # Add verse sorting key for chronological order
            df['verse_sort_key'] = df['Shloka/Verse Number'].apply(parse_verse_number_for_sorting)
            
            df = df.sort_values(['sort_order', 'Sarga/Chapter', 'verse_sort_key'])
            df = df.drop(['sort_order', 'verse_sort_key'], axis=1)
    
    try:
        # Create Excel writer object for multiple sheets
        with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
            # Write main data to 'Verses' sheet
            df.to_excel(writer, sheet_name='Verses', index=False)
            
            # Create and write summary sheet
            summary_df = create_summary_sheet(df)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # If multiple books, create separate sheets for each book
            if 'Kanda/Book' in df.columns and df['Kanda/Book'].nunique() > 1:
                for book in df['Kanda/Book'].unique():
                    if book and book != 'Unknown Kanda':
                        book_df = df[df['Kanda/Book'] == book].copy()
                        # Create a safe sheet name (Excel has 31 character limit)
                        sheet_name = book.replace(' Kanda', '').replace(' ', '_')[:31]
                        book_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"\n=== Success! ===")
        print(f"Combined data saved to: '{output_file_path}'")
        print(f"Total verses processed: {len(df)}")
        print(f"Verses are ordered chronologically as they appear in the text.")
        
        # Display summary statistics
        if 'Kanda/Book' in df.columns:
            print(f"\nBreakdown by Kanda:")
            book_counts = df.groupby('Kanda/Book').size()
            # Maintain chronological order in summary too
            for book in df['Kanda/Book'].unique():
                if book in book_counts:
                    print(f"  - {book}: {book_counts[book]} verses")
        
        # Display sample data
        if len(df) > 0:
            print(f"\nSample data (first 3 rows in chronological order):")
            print(df.head(3).to_string(index=False, max_colwidth=50))
            
        print(f"\nExcel file contains multiple sheets:")
        print(f"  - 'Verses': All combined data (chronologically ordered)")
        print(f"  - 'Summary': Statistics and totals")
        if 'Kanda/Book' in df.columns and df['Kanda/Book'].nunique() > 1:
            print(f"  - Individual sheets for each Kanda")
            
    except Exception as e:
        print(f"Error writing to Excel file: {e}")
        print("Please ensure you have 'openpyxl' installed: pip install openpyxl")

def create_summary_sheet(df):
    """
    Creates a summary DataFrame with statistics for the parsed verses.
    Maintains chronological order of Kandas.
    """
    summary = []
    if 'Kanda/Book' in df.columns:
        # Get counts in chronological order as they appear in the data
        kanda_order = []
        seen_kandas = set()
        for kanda in df['Kanda/Book']:
            if kanda not in seen_kandas:
                kanda_order.append(kanda)
                seen_kandas.add(kanda)
        
        kanda_counts = df['Kanda/Book'].value_counts()
        for kanda in kanda_order:
            if kanda in kanda_counts:
                summary.append({'Kanda/Book': kanda, 'Total Verses': kanda_counts[kanda]})
    
    total_verses = len(df)
    summary.append({'Kanda/Book': 'TOTAL', 'Total Verses': total_verses})
    return pd.DataFrame(summary)

if __name__ == "__main__":
    main()