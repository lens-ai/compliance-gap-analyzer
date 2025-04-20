import os
import sys
import json

# Add app directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models.database import init_mongodb

def main():
    # Load configuration
    with open("app/config.json", "r") as f:
        config = json.load(f)
    
    # Initialize MongoDB
    db = init_mongodb(config["mongodb_uri"], config["mongodb_db"])
    
    print(f"MongoDB collections initialized in database {config['mongodb_db']}:")
    print("\n".join(db.list_collection_names()))
    
    # Create indexes if needed
    print("\nCreating indexes...")
    
    # Example index creation
    db.gapAnalysisResults.create_index([("projectId", 1), ("analysisType", 1)])
    db.gapAnalysisResults.create_index([("moduleId", 1)])
    db.gapAnalysisResults.create_index([("timestamp", -1)])
    
    print("Database initialization complete!")

if __name__ == "__main__":
    main()
