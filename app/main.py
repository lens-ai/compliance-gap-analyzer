import os
import argparse
import json
from compliance_system import ComplianceGapAnalysisSystem

def main():
    parser = argparse.ArgumentParser(description="AI Regulatory Compliance Gap Analysis System")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--project", type=str, help="Jira project key to analyze")
    parser.add_argument("--analysis-type", type=str, default="comprehensive", 
                      choices=["jira", "document", "comprehensive"], 
                      help="Type of analysis to perform")
    parser.add_argument("--documents", type=str, nargs="+", help="Document paths to analyze")
    parser.add_argument("--api", action="store_true", help="Start API server")
    parser.add_argument("--dashboard", action="store_true", help="Launch dashboard")
    
    args = parser.parse_args()
    
    # Initialize the system
    system = ComplianceGapAnalysisSystem(args.config)
    
    if args.api:
        # Import and start Flask API
        from api import app
        app.config["gap_analyzer"] = system
        app.run(host="0.0.0.0", port=5000)
    
    elif args.dashboard:
        # Import and run dashboard
        from dashboard import app as dash_app
        dash_app.config["gap_analyzer"] = system
        dash_app.run_server(debug=True, host="0.0.0.0", port=8050)
    
    else:
        if not args.project:
            parser.error("--project is required when not running API or dashboard")
        
        # Run analysis
        if args.analysis_type == "jira":
            result = system.analyze_jira_tasks(args.project)
        elif args.analysis_type == "document":
            result = system.analyze_documents(args.project, args.documents)
        else:
            result = system.analyze_comprehensive(args.project, args.documents)
        
        print(f"Analysis completed: {result}")

if __name__ == "__main__":
    main()
