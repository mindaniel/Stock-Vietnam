import os
import requests
import zipfile
import io
import shutil

# ==========================================
# ⚙️ SETTINGS
# ==========================================
GITHUB_REPO = "mindaniel/Stock-Vietnam" 
GITHUB_BRANCH = "main"
LOCAL_DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")
LOCAL_VNINDEX_FILE = os.path.join(os.path.dirname(__file__), "VNINDEX.csv")
LOCAL_PUTTHROUGH_FOLDER = os.path.join(os.path.dirname(__file__), "putthrough")
LOCAL_TUDOANH_FOLDER = os.path.join(os.path.dirname(__file__), "tudoanh")

# ==========================================
# 🚀 FAST SYNC SCRIPT (ZIP METHOD)
# ==========================================
def sync_data_fast():
    print(f"🚀 STARTING FAST UPDATE: {GITHUB_REPO}...")
    
    # 1. URL for the "Download ZIP" button
    zip_url = f"https://github.com/{GITHUB_REPO}/archive/refs/heads/{GITHUB_BRANCH}.zip"
    
    try:
        # 2. Download the ZIP file
        print(f"⬇️ Downloading repository archive (this is fast)...")
        response = requests.get(zip_url, timeout=60)
        
        if response.status_code != 200:
            print(f"❌ Failed to download zip: Status {response.status_code}")
            return

        # 3. Process the ZIP in memory
        print(f"📦 Extracting files...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # The zip usually has a root folder like "Stock-Vietnam-main"
            # We need to find where the 'data/' folder is inside it
            all_files = z.namelist()
            
            # Find files that live inside the 'data' folder
            data_files = [f for f in all_files if '/data/' in f and f.endswith('.csv')]
            
            # Find files in tudoanh folder
            tudoanh_files = [f for f in all_files if '/tudoanh/' in f and f.endswith('.csv')]
            
            # Find files in putthrough folder
            putthrough_files = [f for f in all_files if '/putthrough/' in f and f.endswith('.csv')]
            
            # Also find VNINDEX.csv at the root of the repo
            vnindex_file = [f for f in all_files if f.endswith('/VNINDEX.csv') and '/data/' not in f and '/tudoanh/' not in f and '/putthrough/' not in f]
            
            if not data_files and not vnindex_file and not tudoanh_files and not putthrough_files:
                print("⚠️ No data files found in the archive.")
                return

            print(f"📂 Found {len(data_files)} CSV files in data folder.")
            print(f"📂 Found {len(tudoanh_files)} CSV files in tudoanh folder.")
            print(f"📂 Found {len(putthrough_files)} CSV files in putthrough folder.")
            if vnindex_file:
                print(f"📄 Found VNINDEX.csv at repository root.")
            
            if not os.path.exists(LOCAL_DATA_FOLDER):
                os.makedirs(LOCAL_DATA_FOLDER)

            count = 0
            # Extract data folder CSV files
            for file_in_zip in data_files:
                # Get the filename (ignore the folder structure inside zip)
                filename = os.path.basename(file_in_zip)
                
                # Special handling for putthrough_hose_all.csv and tudoanh_all.csv
                if filename == "putthrough_hose_all.csv":
                    if not os.path.exists(LOCAL_PUTTHROUGH_FOLDER):
                        os.makedirs(LOCAL_PUTTHROUGH_FOLDER)
                    target_path = os.path.join(LOCAL_PUTTHROUGH_FOLDER, filename)
                elif filename == "tudoanh_all.csv":
                    if not os.path.exists(LOCAL_TUDOANH_FOLDER):
                        os.makedirs(LOCAL_TUDOANH_FOLDER)
                    target_path = os.path.join(LOCAL_TUDOANH_FOLDER, filename)
                else:
                    target_path = os.path.join(LOCAL_DATA_FOLDER, filename)
                
                # Extract specific file to target
                with z.open(file_in_zip) as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                
                count += 1
                if count % 100 == 0: print(f"   Processed {count} files...", end='\r')
            
            # Extract tudoanh files
            if tudoanh_files:
                if not os.path.exists(LOCAL_TUDOANH_FOLDER):
                    os.makedirs(LOCAL_TUDOANH_FOLDER)
                for file_in_zip in tudoanh_files:
                    filename = os.path.basename(file_in_zip)
                    target_path = os.path.join(LOCAL_TUDOANH_FOLDER, filename)
                    with z.open(file_in_zip) as source, open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target)
                    print(f"   ✅ Updated tudoanh/{filename}")
                    count += 1
            
            # Extract putthrough files
            if putthrough_files:
                if not os.path.exists(LOCAL_PUTTHROUGH_FOLDER):
                    os.makedirs(LOCAL_PUTTHROUGH_FOLDER)
                for file_in_zip in putthrough_files:
                    filename = os.path.basename(file_in_zip)
                    target_path = os.path.join(LOCAL_PUTTHROUGH_FOLDER, filename)
                    with z.open(file_in_zip) as source, open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target)
                    print(f"   ✅ Updated putthrough/{filename}")
                    count += 1
            
            # Extract VNINDEX.csv if found
            if vnindex_file:
                with z.open(vnindex_file[0]) as source, open(LOCAL_VNINDEX_FILE, "wb") as target:
                    shutil.copyfileobj(source, target)
                print(f"   ✅ Updated VNINDEX.csv")

    except Exception as e:
        print(f"❌ Critical Error: {e}")
        return

    print(f"\n✅ SUCCESS! Updated {count} files.")
    print("==========================================")

if __name__ == "__main__":
    sync_data_fast()
