from flask import Flask, request, jsonify, render_template
import threading
import os

app = Flask(__name__)
app.config["gap_analyzer"] = None  # Will be set in main.py

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_project():
    """API endpoint to trigger analysis"""
    gap_analyzer = app.config.get("gap_analyzer")
    if not gap_analyzer:
        return jsonify({"error": "Gap analyzer not initialized"}), 500
    
    data = request.json
    
    if not data or 'project_key' not in data:
        return jsonify({"error": "Missing project_key"}), 400
    
    project_key = data['project_key']
    analysis_type = data.get('analysis_type', 'comprehensive')
    
    # Start analysis in a background thread
    def run_analysis():
        try:
            if analysis_type == "jira":
                gap_analyzer.analyze_jira_tasks(project_key)
            elif analysis_type == "document":
                gap_analyzer.analyze_documents(project_key)
            else:
                gap_analyzer.analyze_comprehensive(project_key)
        except Exception as e:
            print(f"Error in analysis: {e}")
    
    thread = threading.Thread(target=run_analysis)
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "Analysis started", "project_key": project_key, "analysis_type": analysis_type})

@app.route('/api/projects', methods=['GET'])
def get_projects():
    """API endpoint to get list of projects"""
    gap_analyzer = app.config.get("gap_analyzer")
    if not gap_analyzer:
        return jsonify({"error": "Gap analyzer not initialized"}), 500
    
    # Get projects from Jira
    try:
        projects = gap_analyzer.jira_connector.get_projects()
        return jsonify({"projects": [{"key": p["key"], "name": p["name"]} for p in projects]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/results/<project_key>', methods=['GET'])
def get_results(project_key):
    """API endpoint to get analysis results"""
    gap_analyzer = app.config.get("gap_analyzer")
    if not gap_analyzer:
        return jsonify({"error": "Gap analyzer not initialized"}), 500
    
    # Get analysis type from query param
    analysis_type = request.args.get('analysis_type', 'comprehensive')
    
    # Query MongoDB for results
    db = gap_analyzer.db
    query = {"projectId": project_key, "analysisType": analysis_type}
    
    try:
        results = list(db.gapAnalysisResults.find(query))
        # Convert ObjectId to string for JSON serialization
        for result in results:
            result["_id"] = str(result["_id"])
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload-document', methods=['POST'])
def upload_document():
    """API endpoint to upload a document for analysis"""
    gap_analyzer = app.config.get("gap_analyzer")
    if not gap_analyzer:
        return jsonify({"error": "Gap analyzer not initialized"}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    project_key = request.form.get('project_key')
    if not project_key:
        return jsonify({"error": "Missing project_key"}), 400
    
    file = request.files['file']
    
    # Save file to a temporary location
    file_path = os.path.join("temp", file.filename)
    os.makedirs("temp", exist_ok=True)
    file.save(file_path)
    
    # Process document
    try:
        result = gap_analyzer.analyze_documents(project_key, [file_path])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)
