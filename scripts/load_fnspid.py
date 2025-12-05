import pandas as pd
import os
import glob

def load_fnspid():
    print("=" * 70)
    print("FNSPID DATASET LOADER")
    print("=" * 70)
    
    # Create output directory
    os.makedirs("data/annotations", exist_ok=True)
    
    possible_base_paths = [
        "data/raw/FNSPID_Financial_News_Dataset",
        "FNSPID_Financial_News_Dataset",
        "data/raw",
        "."
    ]
    
    data_processor_path = None
    for base in possible_base_paths:
        test_path = os.path.join(base, "data_processor")
        if os.path.exists(test_path):
            data_processor_path = test_path
            print(f"\n✓ Found data_processor at: {test_path}")
            break
    
    if not data_processor_path:
        print("\n Could not find data_processor folder!")
        print("\n SETUP INSTRUCTIONS:")
        print("=" * 70)
        print("\nThe Download_dataset.py file is empty, so you need to:")
        print("\n1. Clone the GitHub repository:")
        print("   mkdir -p data/raw")
        print("   cd data/raw")
        print("   git clone https://github.com/Zdong104/FNSPID_Financial_News_Dataset.git")
        print("   cd ../..")
        print("\n2. Run this script again:")
        print("   python scripts/load_fnspid.py")
        print("\nThe script will automatically find and process the data.")
        print("=" * 70)
        return
    
    # Try different data folders in priority order
    data_folders = [
        "news_data_preprocessed",
        "news_data_sentiment_scored", 
        "news_data_raw",
    ]
    
    data_path = None
    for folder in data_folders:
        test_path = os.path.join(data_processor_path, folder)
        if os.path.exists(test_path):
            files = glob.glob(os.path.join(test_path, "*.csv")) + \
                    glob.glob(os.path.join(test_path, "*.parquet")) + \
                    glob.glob(os.path.join(test_path, "*.json"))
            if files:
                data_path = test_path
                print(f" Using data from: {folder}/")
                break
    
    if not data_path:
        print("\n No data files found in data_processor folders!")
        print("\nChecked folders:")
        for folder in data_folders:
            print(f"  • {folder}/")
        return
    
    data_files = glob.glob(os.path.join(data_path, "*.csv")) + \
                 glob.glob(os.path.join(data_path, "*.parquet")) + \
                 glob.glob(os.path.join(data_path, "*.json"))
    
    if not data_files:
        print(f"\n No data files found in: {data_path}")
        return
    
    print(f"\n Found {len(data_files)} data file(s):")
    for i, f in enumerate(data_files[:10], 1):
        print(f"   {i}. {os.path.basename(f)}")
    if len(data_files) > 10:
        print(f"   ... and {len(data_files) - 10} more")
    
    # Load all files
    all_dfs = []
    print(f"\n Loading data files...")
    
    for file_path in data_files:
        filename = os.path.basename(file_path)
        print(f"   Loading: {filename}", end="")
        
        try:
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path, lines=True)
            else:
                # Try CSV with different encodings
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(file_path, encoding='latin-1', low_memory=False)
                    except:
                        df = pd.read_csv(file_path, encoding='utf-8', errors='ignore', low_memory=False)
            
            if len(df) > 0:
                all_dfs.append(df)
                print(f" {len(df):,} rows")
            else:
                print(" (empty, skipped)")
                
        except Exception as e:
            print(f" Error: {str(e)[:50]}")
            continue
    
    if not all_dfs:
        print("\n Failed to load any data files successfully")
        return
    
    # Combine all dataframes
    print(f"\n Combining {len(all_dfs)} file(s)...")
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"✓ Total rows: {len(df):,}")
    
    # Remove duplicates
    original_len = len(df)
    df = df.drop_duplicates()
    if len(df) < original_len:
        print(f"✓ Removed {original_len - len(df):,} duplicate rows")
    
    # Show structure
    print(f"\n Columns: {list(df.columns)}")
    print("\n" + "=" * 70)
    print("SAMPLE DATA:")
    print("=" * 70)
    print(df.head(2))
    
    # Find text column
    text_columns = [
        'Article_title', 'article_title', 'title', 'Title',
        'Article', 'article', 'text', 'Text', 'content',
        'headline', 'Headline', 'sentence', 'news_title'
    ]
    
    sentence_col = None
    for col in text_columns:
        if col in df.columns:
            sentence_col = col
            break
    
    # If still not found, look for string column with long text
    if not sentence_col:
        for col in df.columns:
            if df[col].dtype == 'object':
                sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else ""
                if isinstance(sample, str) and len(sample) > 20:
                    sentence_col = col
                    break
    
    if not sentence_col:
        print("\n Could not identify text column")
        print("Available columns:", list(df.columns))
        return
    
    print(f"\n✓ Using text column: '{sentence_col}'")
    
    # Rename columns
    rename_map = {sentence_col: 'sentence'}
    
    # Add other columns if they exist
    column_mappings = {
        'Date': 'date', 'date': 'date', 'Date_string': 'date',
        'Stock_symbol': 'stock_symbol', 'stock_symbol': 'stock_symbol', 'symbol': 'stock_symbol',
        'Publisher': 'publisher', 'publisher': 'publisher',
        'Url': 'url', 'url': 'url',
    }
    
    for old_col, new_col in column_mappings.items():
        if old_col in df.columns and new_col not in rename_map.values():
            rename_map[old_col] = new_col
    
    df = df.rename(columns=rename_map)
    
    # Hedge detection
    print("\n Analyzing hedge language...")
    
    hedge_keywords = [
        'may', 'might', 'could', 'would', 'possibly', 'perhaps', 
        'likely', 'unlikely', 'appear', 'seem', 'suggest', 'indicate',
        'estimate', 'forecast', 'expect', 'expected', 'project', 'potential',
        'approximately', 'around', 'about', 'roughly', 'believed', 'appears',
        'seems', 'reported', 'reportedly', 'alleged', 'allegedly', 'rumored',
        'speculation', 'speculate', 'anticipated', 'probably', 'presumably',
        'poised', 'set to', 'plans to', 'aims to', 'seeks to', 'intends'
    ]
    
    def contains_hedge(text):
        if pd.isna(text):
            return 0
        text_lower = str(text).lower()
        # Use word boundaries to avoid false positives
        words = set(text_lower.replace(',', ' ').replace('.', ' ').split())
        return int(any(keyword in words for keyword in hedge_keywords))
    
    df["hedge"] = df["sentence"].apply(contains_hedge)
    df["label"] = df["hedge"]
    
    # Select columns to keep
    columns_to_keep = ["sentence", "label", "hedge"]
    for col in ["date", "stock_symbol", "publisher", "url"]:
        if col in df.columns:
            columns_to_keep.append(col)
    
    df = df[columns_to_keep]
    
    # Clean data
    print("🧹 Cleaning data...")
    df = df.dropna(subset=["sentence"])
    df = df[df["sentence"].str.len() > 10]
    df = df.reset_index(drop=True)
    
    # Save
    output_path = "data/annotations/fnspid_clean.csv"
    df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 70)
    print("SUCCESS!")
    print("=" * 70)
    print(f"✓ Saved: {output_path}")
    print(f"✓ Total rows: {len(df):,}")
    print(f"✓ Hedge sentences: {df['hedge'].sum():,} ({df['hedge'].mean()*100:.1f}%)")
    print(f"✓ Non-hedge: {(1-df['hedge']).sum():,} ({(1-df['hedge'].mean())*100:.1f}%)")
    
    # Show examples
    print("\n Sample hedge sentences:")
    hedge_samples = df[df['hedge'] == 1].head(5)
    for i, row in hedge_samples.iterrows():
        text = row['sentence'][:80] + "..." if len(row['sentence']) > 80 else row['sentence']
        print(f"   • {text}")
    
    print("\n Sample non-hedge sentences:")
    non_hedge_samples = df[df['hedge'] == 0].head(5)
    for i, row in non_hedge_samples.iterrows():
        text = row['sentence'][:80] + "..." if len(row['sentence']) > 80 else row['sentence']
        print(f"   • {text}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    load_fnspid()
    