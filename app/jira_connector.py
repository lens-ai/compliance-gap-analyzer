import requests
import json
import base64
from typing import Dict, List, Optional, Any, BinaryIO
import os

class EnhancedJiraConnector:
    def __init__(self, base_url: str, email: str, api_token: str):
        self.base_url = base_url
        self.auth_header = {
            "Authorization": f"Basic {base64.b64encode(f'{email}:{api_token}'.encode()).decode()}",
            "Content-Type": "application/json"
        }
    
    def get_projects(self) -> List[Dict]:
        """Get all projects in Jira"""
        url = f"{self.base_url}/rest/api/3/project"
        response = requests.get(url, headers=self.auth_header)
        response.raise_for_status()
        return response.json()
    
    def extract_project_data(self, project_key: str) -> Dict:
        """Extract all relevant data for a project"""
        # Get project info
        project = self.get_project(project_key)
        
        # Get epics
        epics = self.get_epics(project_key)
        
        # Process epics, features and stories
        processed_epics = []
        for epic in epics:
            epic_id = epic["key"]
            epic_data = {
                "_id": epic_id,
                "title": epic["fields"].get("summary", ""),
                "description": epic["fields"].get("description", ""),
                "status": epic["fields"]["status"]["name"],
                "features": []
            }
            
            # Get features (implemented as stories in Jira with epic link)
            features = self.get_stories_by_epic(epic_id)
            
            # Group stories by feature
            feature_groups = self._group_stories_by_feature(features)
            
            # Process features
            for feature_name, stories in feature_groups.items():
                feature_id = f"{epic_id}_{feature_name.replace(' ', '_').lower()}"
                feature_data = {
                    "_id": feature_id,
                    "title": feature_name,
                    "description": f"Feature: {feature_name}",
                    "status": self._derive_status_from_stories(stories),
                    "stories": []
                }
                
                # Process stories
                for story in stories:
                    story_id = story["key"]
                    story_data = {
                        "_id": story_id,
                        "title": story["fields"].get("summary", ""),
                        "description": story["fields"].get("description", ""),
                        "storyFormat": self._extract_user_story_format(story),
                        "status": story["fields"]["status"]["name"],
                        "assignedTo": story["fields"].get("assignee", {}).get("accountId", None),
                        "tasks": []
                    }
                    
                    # Get tasks (subtasks in Jira)
                    subtasks = self.get_subtasks(story_id)
                    for subtask in subtasks:
                        task_id = subtask["key"]
                        task_data = {
                            "_id": task_id,
                            "title": subtask["fields"].get("summary", ""),
                            "description": subtask["fields"].get("description", ""),
                            "status": subtask["fields"]["status"]["name"],
                            "assignedTo": subtask["fields"].get("assignee", {}).get("accountId", None)
                        }
                        story_data["tasks"].append(task_data)
                    
                    feature_data["stories"].append(story_data)
                
                epic_data["features"].append(feature_data)
            
            processed_epics.append(epic_data)
        
        return {
            "_id": f"project-{project_key.lower()}",
            "projectName": project["name"],
            "description": project.get("description", ""),
            "epics": processed_epics
        }
    
    def get_project(self, project_key: str) -> Dict:
        """Get project information by key"""
        url = f"{self.base_url}/rest/api/3/project/{project_key}"
        response = requests.get(url, headers=self.auth_header)
        response.raise_for_status()
        return response.json()
    
    def get_epics(self, project_key: str) -> List[Dict]:
        """Get all epics in a project"""
        jql = f"project = {project_key} AND issuetype = Epic"
        fields = ["summary", "description", "status"]
        return self.get_all_issues(jql, fields)
    
    def get_stories_by_epic(self, epic_key: str) -> List[Dict]:
        """Get all stories in an epic"""
        jql = f"'Epic Link' = {epic_key}"
        fields = ["summary", "description", "status", "assignee", "labels", "components"]
        return self.get_all_issues(jql, fields)
    
    def get_subtasks(self, parent_key: str) -> List[Dict]:
        """Get all subtasks for a parent issue"""
        jql = f"parent = {parent_key}"
        fields = ["summary", "description", "status", "assignee"]
        return self.get_all_issues(jql, fields)
    
    def get_all_issues(self, jql: str, fields: List[str]) -> List[Dict]:
        """Get all issues matching JQL query"""
        all_issues = []
        start_at = 0
        max_results = 100
        total = 1  # Initial value to enter the loop
        
        while start_at < total:
            response = self.get_issues(jql, fields, start_at, max_results)
            total = response['total']
            all_issues.extend(response['issues'])
            start_at += max_results
            
        return all_issues
    
    def get_issues(self, jql: str, fields: List[str], start_at: int = 0, max_results: int = 100) -> Dict:
        """Search for issues using JQL"""
        url = f"{self.base_url}/rest/api/3/search"
        payload = {
            "jql": jql,
            "startAt": start_at,
            "maxResults": max_results,
            "fields": fields
        }
        response = requests.post(url, headers=self.auth_header, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()
    
    def get_project_attachments(self, project_key: str) -> List[Dict]:
        """Get all attachments in a project"""
        # First get all issues with attachments
        jql = f"project = {project_key} AND attachments IS NOT EMPTY"
        fields = ["attachment"]
        issues = self.get_all_issues(jql, fields)
        
        # Extract attachments
        attachments = []
        for issue in issues:
            for attachment in issue["fields"].get("attachment", []):
                attachments.append({
                    "id": attachment["id"],
                    "filename": attachment["filename"],
                    "content_type": attachment["mimeType"],
                    "size": attachment["size"],
                    "url": attachment["content"],
                    "issue_key": issue["key"]
                })
        
        return attachments
    
    def download_attachment(self, attachment_id: str, file_path: str) -> bool:
        """Download an attachment to a file"""
        url = f"{self.base_url}/rest/api/3/attachment/content/{attachment_id}"
        
        try:
            response = requests.get(url, headers=self.auth_header, stream=True)
            response.raise_for_status()
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
        
        except Exception as e:
            print(f"Error downloading attachment {attachment_id}: {e}")
            return False
    
    def create_issue(self, project_key: str, issue_type: str, summary: str, description: str, 
                    additional_fields: Optional[Dict] = None) -> Dict:
        """Create a new issue"""
        url = f"{self.base_url}/rest/api/3/issue"
        fields = {
            "project": {"key": project_key},
            "issuetype": {"name": issue_type},
            "summary": summary,
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [
                            {
                                "type": "text",
                                "text": description
                            }
                        ]
                    }
                ]
            }
        }
        
        if additional_fields:
            fields.update(additional_fields)
            
        payload = {"fields": fields}
        response = requests.post(url, headers=self.auth_header, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()
    
    def add_comment(self, issue_key: str, comment: str) -> Dict:
        """Add a comment to an issue"""
        url = f"{self.base_url}/rest/api/3/issue/{issue_key}/comment"
        
        payload = {
            "body": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [
                            {
                                "type": "text",
                                "text": comment
                            }
                        ]
                    }
                ]
            }
        }
        
        response = requests.post(url, headers=self.auth_header, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()
    
    def _group_stories_by_feature(self, stories: List[Dict]) -> Dict[str, List[Dict]]:
        """Group stories by feature based on labels or naming patterns"""
        feature_groups = {}
        
        for story in stories:
            # Try to extract feature from labels
            labels = story["fields"].get("labels", [])
            feature_label = next((label for label in labels if label.startswith("feature-")), None)
            
            if feature_label:
                feature_name = feature_label.replace("feature-", "").replace("-", " ").title()
            else:
                # Try to extract from summary using pattern matching
                summary = story["fields"].get("summary", "")
                if ":" in summary:
                    feature_name = summary.split(":")[0].strip()
                else:
                    # Default to "General" if no feature identifier found
                    feature_name = "General"
            
            if feature_name not in feature_groups:
                feature_groups[feature_name] = []
            
            feature_groups[feature_name].append(story)
        
        return feature_groups
    
    def _derive_status_from_stories(self, stories: List[Dict]) -> str:
        """Derive feature status from story statuses"""
        if not stories:
            return "Unknown"
        
        statuses = [story["fields"]["status"]["name"] for story in stories]
        
        if all(status == "Done" for status in statuses):
            return "Completed"
        elif all(status == "To Do" for status in statuses):
            return "Not Started"
        else:
            return "In Progress"
    
    def _extract_user_story_format(self, story: Dict) -> str:
        """Extract user story format from description or construct from summary"""
        description = story["fields"].get("description", "")
        summary = story["fields"].get("summary", "")
        
        # Check if description contains user story format
        if description and ("As a " in description or "As an " in description):
            return description
        
        # Try to extract from summary
        if "As a " in summary or "As an " in summary:
            return summary
        
        # Construct generic format
        return f"As a user, I need {summary} so that I can achieve the related business goal."
