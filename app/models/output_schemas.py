from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class JiraTaskAnalysisResult(BaseModel):
    aligned: bool = Field(description="Whether the task aligns with the requirement")
    explanation: str = Field(description="Detailed explanation of the alignment or gaps")
    gaps: List[str] = Field(description="List of identified gaps")
    suggestions: List[str] = Field(description="List of suggested improvements")
    score: int = Field(description="Compliance score (0-100)")
    jira_updates: List[Dict] = Field(description="Suggested Jira updates")

class DocumentAnalysisResult(BaseModel):
    aligned: bool = Field(description="Whether the document aligns with the template")
    explanation: str = Field(description="Detailed explanation of the alignment or gaps")
    missing_sections: List[str] = Field(description="List of missing sections")
    content_gaps: List[str] = Field(description="List of content gaps")
    improvement_suggestions: List[str] = Field(description="List of suggested improvements")
    tasks_to_create: List[str] = Field(description="List of tasks to create")
    score: int = Field(description="Compliance score (0-100)")

class ComprehensiveAnalysisResult(BaseModel):
    aligned: bool = Field(description="Whether the implementation aligns with the requirement")
    explanation: str = Field(description="Detailed explanation of the alignment or gaps")
    task_gaps: List[str] = Field(description="List of task implementation gaps")
    documentation_gaps: List[str] = Field(description="List of documentation gaps")
    suggested_tasks: List[str] = Field(description="List of suggested tasks")
    suggested_document_updates: List[str] = Field(description="List of suggested document updates")
    score: int = Field(description="Compliance score (0-100)")
