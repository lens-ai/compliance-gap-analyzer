import pymongo
from datetime import datetime

def init_mongodb(mongo_uri, db_name):
    """Initialize MongoDB connection and collections"""
    # Connect to master DB (Atlas)
    master_client = pymongo.MongoClient(mongo_uri_config["master"])
    master_db = master_client[db_name_config["master"]]
    
    # Connect to local DB
    local_client = pymongo.MongoClient(mongo_uri_config["local"])
    local_db = local_client[db_name_config["local"]]
     
    # Create collections if they don't exist
    if "gapAnalysisResults" not in db.list_collection_names():
        local_db.create_collection("gapAnalysisResults")
        # Create indexes for better query performance
        local_db.gapAnalysisResults.create_index([("projectId", pymongo.ASCENDING)])
        local_db.gapAnalysisResults.create_index([("analysisType", pymongo.ASCENDING)])
        local_db.gapAnalysisResults.create_index([("timestamp", pymongo.DESCENDING)])
    
    if "gapAnalysisSummary" not in db.list_collection_names():
        local_db.create_collection("gapAnalysisSummary")
        local_db.gapAnalysisSummary.create_index([("projectId", pymongo.ASCENDING)], unique=True)
    
    if "resourceAndTrainingNeeds" not in db.list_collection_names():
        local_db.create_collection("resourceAndTrainingNeeds")
        local_db.resourceAndTrainingNeeds.create_index([
            ("projectId", pymongo.ASCENDING),
            ("moduleId", pymongo.ASCENDING)
        ], unique=True)
    
    if "gapAnalysisHistory" not in db.list_collection_names():
        local_db.create_collection("gapAnalysisHistory")
        local_db.gapAnalysisHistory.create_index([
            ("projectId", pymongo.ASCENDING),
            ("timestamp", pymongo.DESCENDING)
        ])
    
    return {
            "master": master_db,
            "local": local_db
    }
