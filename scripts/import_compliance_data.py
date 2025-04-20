import os
import sys
import json
from datetime import datetime

# Add app directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    # Load configuration
    with open("app/config.json", "r") as f:
        config = json.load(f)
    
    # Import compliance data
    compliance_db_path = config.get("compliance_db_path", "data/compliance_db.json")
    
    # Check if the file exists
    if not os.path.exists(compliance_db_path):
        print(f"Compliance database file not found at {compliance_db_path}")
        
        # Check if data directory exists, create if not
        data_dir = os.path.dirname(compliance_db_path)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Create a minimal placeholder file
        with open(compliance_db_path, "w") as f:
            json.dump({
                "frameworks": [],
                "regulatoryArticles": [],
                "complianceRequirements": [],
                "documentTemplates": []
            }, f)
        
        print(f"Created empty compliance database file at {compliance_db_path}")
        print("Please populate it with your compliance data")
        return
    
    print(f"Using compliance database from {compliance_db_path}")
    with open(compliance_db_path, "r") as f:
        compliance_data = json.load(f)
    
    print(f"Found {len(compliance_data.get('frameworks', []))} frameworks")
    print(f"Found {len(compliance_data.get('regulatoryArticles', []))} articles")
    print(f"Found {len(compliance_data.get('complianceRequirements', []))} requirements")
    print(f"Found {len(compliance_data.get('documentTemplates', []))} templates")
    
    print("\nCompliance data is ready for use by the application.")

if __name__ == "__main__":
    main()
