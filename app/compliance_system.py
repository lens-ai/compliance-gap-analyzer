import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import pymongo
import json
import requests
import re

# LangChain imports
from langchain.document_loaders import (
    PyPDFLoader, 
    UnstructuredWordDocumentLoader, 
    ConfluenceLoader, 
    DirectoryLoader
)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import Document

# Import Pydantic models for structured output
from models.output_schemas import (
    JiraTaskAnalysisResult,
    DocumentAnalysisResult,
    ComprehensiveAnalysisResult
)

class ComplianceGapAnalysisSystem:
    def __init__(self, config_path):
        """Initialize the compliance gap analysis system with LangChain components"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Set up logging
        self._setup_logging()
        
        # Initialize MongoDB
        # Initialize MongoDB connections
        db_connections = init_mongodb(
            self.config["mongodb_uri"],
            self.config["mongodb_db"]
        )

        # Set up DB references
        self.master_db = db_connections["master"]
        self.db = db_connections["local"]  # This is the local DB for gap analysis results

        # Initialize LangChain components
        self.embeddings = self._init_embeddings()
        self.text_splitter = self._init_text_splitter()
        self.vector_store = self._init_vector_store()
        
        # Initialize helper components
        self.jira_connector = self._init_jira_connector()
        self.document_processor = self._init_document_processor()
        
        # Initialize compliance DB loader
        self.compliance_db = self._init_compliance_db()
        
        # Initialize vector store with compliance data
        self._initialize_vector_store()
        
        # Initialize analysis chains
        self.analysis_chains = self._init_analysis_chains()
    
    def _setup_logging(self):
        """Set up logging configuration"""
        log_dir = os.path.dirname(self.config.get("log_file", "logs/compliance_gap.log"))
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.get("log_file", "logs/compliance_gap.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Compliance Gap Analysis System")
    
    def _init_embeddings(self):
        """Initialize embeddings model"""
        return HuggingFaceEmbeddings(
            model_name=self.config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        )
    
    def _init_text_splitter(self):
        """Initialize text splitter"""
        return RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def _init_vector_store(self):
        """Initialize vector store"""
        vector_store_dir = self.config.get("vector_store_dir", "vector_store")
        os.makedirs(vector_store_dir, exist_ok=True)
        
        # Use FAISS for fast similarity search
        try:
            return FAISS.load_local(vector_store_dir, self.embeddings)
        except:
            # Create new vector store if none exists
            return FAISS.from_texts(
                texts=["Placeholder"], 
                embedding=self.embeddings,
                persist_directory=vector_store_dir
            )
    
    def _init_jira_connector(self):
        """Initialize Jira connector"""
        # Import here to avoid circular imports
        from jira_connector import EnhancedJiraConnector
        
        return EnhancedJiraConnector(
            self.config["jira"]["base_url"],
            self.config["jira"]["email"],
            self.config["jira"]["api_token"]
        )
    
    def _init_document_processor(self):
        """Initialize document processor with LangChain components"""
        return {
            "pdf": PyPDFLoader,
            "docx": UnstructuredWordDocumentLoader,
            "confluence": lambda url: ConfluenceLoader(
                url=self.config["confluence"]["base_url"],
                username=self.config["confluence"]["username"],
                api_key=self.config["confluence"]["api_token"]
            )
        }
    
    def _init_compliance_db(self):
        """Load compliance database"""
        compliance_docs = []
         
        try:
            # Get frameworks from master DB
            frameworks = list(self.master_db.frameworks.find({}))
            for framework in frameworks:
                doc = Document(
                    page_content=f"{framework.get('name')}: {framework.get('description')}",
                    metadata={
                        "type": "framework",
                        "id": framework.get("id"),
                        "name": framework.get("name"),
                        "shortCode": framework.get("shortCode")
                    }
                )
                compliance_docs.append(doc)
            
            articles = list(self.master_db.regulatoryArticles.find({}))
            for article in articles:
                doc = Document(
                    page_content=f"{article.get('title')}: {article.get('text')}",
                    metadata={
                        "type": "article",
                        "id": article.get("_id"),
                        "frameworkId": article.get("frameworkId"),
                        "articleNumber": article.get("articleNumber"),
                        "title": article.get("title")
                    }
                )
                compliance_docs.append(doc)
                
                # Add sub-articles
                for sub_article in article.get("subArticles", []):
                    sub_doc = Document(
                        page_content=f"{sub_article.get('title')}: {sub_article.get('text')}",
                        metadata={
                            "type": "sub_article",
                            "id": sub_article.get("_id"),
                            "parent_id": article.get("_id"),
                            "frameworkId": article.get("frameworkId"),
                            "articleNumber": sub_article.get("articleNumber"),
                            "title": sub_article.get("title")
                        }
                    )
                    compliance_docs.append(sub_doc)

            compliance = list(self.master_db.complianceRequirements.find({}))
            for req in compliance_data.get("complianceRequirements", []):
                doc = Document(
                    page_content=f"{req.get('taskName')}: {req.get('description')}",
                    metadata={
                        "type": "requirement",
                        "id": req.get("_id"),
                        "taskName": req.get("taskName"),
                        "category": req.get("category"),
                        "importance": req.get("importance")
                    }
                )
                compliance_docs.append(doc)
            
            # Load document templates if available
            template_path = self.config.get("document_templates_path")
            if template_path and os.path.exists(template_path):
                try:
                    loader = DirectoryLoader(
                        template_path, 
                        glob="**/*.*",
                        loader_cls=UnstructuredWordDocumentLoader
                    )
                    template_docs = loader.load()
                    
                    # Add metadata to templates
                    for doc in template_docs:
                        doc.metadata.update({
                            "type": "template",
                            "id": os.path.basename(doc.metadata.get("source", "")),
                            "title": os.path.basename(doc.metadata.get("source", "")).split(".")[0]
                        })
                    
                    compliance_docs.extend(template_docs)
                except Exception as e:
                    self.logger.error(f"Error loading templates: {e}")
        
        except Exception as e:
            self.logger.error(f"Error loading compliance database: {e}")
        
        return compliance_docs
    
    def _initialize_vector_store(self):
        """Initialize vector store with compliance documents"""
        if not self.compliance_db:
            self.logger.warning("No compliance documents to add to vector store")
            return
        
        try:
            # Split compliance documents
            texts = []
            metadatas = []
            
            for doc in self.compliance_db:
                # Split document if needed
                if len(doc.page_content) > 1000:
                    splits = self.text_splitter.split_documents([doc])
                    for split in splits:
                        texts.append(split.page_content)
                        metadatas.append(split.metadata)
                else:
                    texts.append(doc.page_content)
                    metadatas.append(doc.metadata)
            
            # Create new vector store
            self.vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            
            # Save to disk
            vector_store_dir = self.config.get("vector_store_dir", "vector_store")
            self.vector_store.save_local(vector_store_dir)
            
            self.logger.info(f"Vector store initialized with {len(texts)} compliance documents")
        
        except Exception as e:
            self.logger.error(f"Error initializing vector store: {e}")
    
    def _init_analysis_chains(self):
        """Initialize analysis chains with DeepSeek LLM"""
        # Create output parsers
        jira_parser = PydanticOutputParser(pydantic_object=JiraTaskAnalysisResult)
        document_parser = PydanticOutputParser(pydantic_object=DocumentAnalysisResult)
        comprehensive_parser = PydanticOutputParser(pydantic_object=ComprehensiveAnalysisResult)
        
        # Create prompt templates for each analysis type
        jira_prompt = ChatPromptTemplate.from_template("""
        You are an AI regulatory compliance expert specialized in analyzing gaps between 
        implementation tasks and regulatory requirements. Your task is to analyze if the client's 
        implementation task aligns with the specific compliance requirement.
        
        ## COMPLIANCE REQUIREMENT
        Framework: {framework_id}
        Article: {article_id} - {article_title}
        Content: {article_content}
        
        ## CLIENT IMPLEMENTATION
        Title: {task_title}
        Description: {task_description}
        User Story: {user_story}
        Status: {task_status}
        
        Tasks:
        {tasks}
        
        ## ANALYSIS TASK
        1. Is the client implementation aligned with the compliance requirement? Provide a detailed explanation.
        2. Identify specific gaps or missing elements in the client implementation.
        3. Suggest specific improvements or new tasks to address the gaps.
        4. Provide a compliance score from 0-100 based on how well the implementation meets the requirement.
        
        {format_instructions}
        """)
        
        document_prompt = ChatPromptTemplate.from_template("""
        You are an AI regulatory compliance expert specializing in document review. Your task is to analyze if the client's 
        document aligns with required compliance document templates and contains all necessary sections and content.
        
        ## REQUIRED TEMPLATE
        Template Name: {template_name}
        Template Purpose: {template_purpose}
        Required Sections:
        {required_sections}
        
        ## CLIENT DOCUMENT
        Document Title: {document_title}
        Document Type: {document_type}
        Document Sections:
        {document_sections}
        
        ## ANALYSIS TASK
        1. Is the client document aligned with the required template? Provide a detailed explanation.
        2. Identify specific missing sections or content in the client document.
        3. Suggest specific improvements to align the document with compliance requirements.
        4. Provide a compliance score from 0-100 based on how well the document meets the template requirements.
        
        {format_instructions}
        """)
        
        comprehensive_prompt = ChatPromptTemplate.from_template("""
        You are an AI regulatory compliance expert specialized in comprehensive compliance analysis. Your task is to analyze 
        if the client's implementation (both tasks and documents) aligns with regulatory requirements.
        
        ## COMPLIANCE REQUIREMENT
        Framework: {framework_id}
        Article: {article_id} - {article_title}
        Content: {article_content}
        
        ## CLIENT IMPLEMENTATION
        Tasks:
        {tasks}
        
        Documents:
        {documents}
        
        ## ANALYSIS TASK
        1. Analyze how well the client's implementation (both tasks and documents) addresses the compliance requirement.
        2. Identify specific gaps in both task implementation and documentation.
        3. Provide a holistic view of compliance status and recommendations.
        4. Suggest specific tasks to address all identified gaps.
        
        {format_instructions}
        """)
        
        # Add format instructions to prompts
        jira_prompt_with_parser = jira_prompt.partial(
            format_instructions=jira_parser.get_format_instructions()
        )
        
        document_prompt_with_parser = document_prompt.partial(
            format_instructions=document_parser.get_format_instructions()
        )
        
        comprehensive_prompt_with_parser = comprehensive_prompt.partial(
            format_instructions=comprehensive_parser.get_format_instructions()
        )
        
        return {
            "jira": {"prompt": jira_prompt_with_parser, "parser": jira_parser},
            "document": {"prompt": document_prompt_with_parser, "parser": document_parser},
            "comprehensive": {"prompt": comprehensive_prompt_with_parser, "parser": comprehensive_parser},
        }
    
    def invoke_deepseek(self, prompt, temperature=0.2, max_tokens=2000):
        """Call DeepSeek API for LLM analysis"""
        deepseek_api_url = self.config.get("llm", {}).get("api_url", "https://api.deepseek.com/v1/chat/completions")
        api_key = self.config.get("llm", {}).get("api_key", "")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.get("llm", {}).get("model_name", "deepseek-chat"),
            "messages": [
                {"role": "system", "content": "You are an AI regulatory compliance expert."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(deepseek_api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            self.logger.error(f"Error calling DeepSeek API: {e}")
            raise
    
    def process_document(self, file_path):
        """Process a document with LangChain"""
        # Determine file type
        _, ext = os.path.splitext(file_path)
        doc_type = ext.lstrip('.').lower()
        
        # Get appropriate loader
        if doc_type == "pdf":
            loader = self.document_processor["pdf"]
        elif doc_type in ["docx", "doc"]:
            loader = self.document_processor["docx"]
        else:
            raise ValueError(f"Unsupported document type: {doc_type}")
        
        # Load and process document
        try:
            loader_instance = loader(file_path)
            docs = loader_instance.load()
            
            # Extract metadata
            metadata = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "document_type": doc_type
            }
            
            # Extract sections (simplified)
            sections = []
            current_heading = None
            current_content = ""
            
            for doc in docs:
                # Very basic section extraction - in a real implementation, 
                # you'd want more sophisticated parsing
                lines = doc.page_content.split("\n")
                for line in lines:
                    # Simple heuristic for heading detection
                    if line.strip() and (len(line) < 100 and line.strip().isupper() or 
                                         line.strip().startswith("#") or
                                         bool(re.match(r"^\d+\.\s+", line))):
                        # Save previous section
                        if current_heading:
                            sections.append({
                                "title": current_heading,
                                "content": current_content
                            })
                        
                        # Start new section
                        current_heading = line.strip()
                        current_content = ""
                    else:
                        current_content += line + "\n"
            
            # Add the last section
            if current_heading:
                sections.append({
                    "title": current_heading,
                    "content": current_content
                })
            
            # If no sections were found, create a default one
            if not sections:
                sections.append({
                    "title": "Document Content",
                    "content": docs[0].page_content if docs else ""
                })
            
            return {
                "docs": docs,
                "metadata": metadata,
                "sections": sections,
                "content": "\n\n".join([doc.page_content for doc in docs])
            }
        
        except Exception as e:
            self.logger.error(f"Error processing document {file_path}: {e}")
            raise
    
    def analyze_jira_tasks(self, project_key):
        """Analyze Jira tasks against compliance requirements"""
        self.logger.info(f"Starting Jira task analysis for project {project_key}")
        
        # Extract project data from Jira
        project_data = self.jira_connector.extract_project_data(project_key)
        
        # Analyze gaps
        results = []
        
        # Process epics, features, and stories
        for epic in project_data.get("epics", []):
            for feature in epic.get("features", []):
                for story in feature.get("stories", []):
                    # Create embeddings for the story
                    story_text = f"{story.get('title', '')}\n{story.get('storyFormat', '')}"
                    
                    # Search for relevant compliance items
                    compliance_items = self.vector_store.similarity_search(
                        story_text, 
                        k=3,
                        filter={"type": {"$in": ["article", "sub_article", "requirement"]}}
                    )
                    
                    if not compliance_items:
                        continue
                    
                    # Analyze against each compliance item
                    for item in compliance_items:
                        try:
                            # Extract compliance info
                            compliance_meta = item.metadata
                            framework_id = compliance_meta.get("frameworkId", "")
                            article_id = compliance_meta.get("id", "")
                            article_title = compliance_meta.get("title", "")
                            
                            # Format tasks text
                            tasks_text = ""
                            for task in story.get("tasks", []):
                                tasks_text += f"- {task.get('title', '')}: {task.get('description', '')}\n"
                            
                            # Run analysis with DeepSeek
                            prompt = self.analysis_chains["jira"]["prompt"].format(
                                framework_id=framework_id,
                                article_id=article_id,
                                article_title=article_title,
                                article_content=item.page_content,
                                task_title=story.get("title", ""),
                                task_description=story.get("description", ""),
                                user_story=story.get("storyFormat", ""),
                                task_status=story.get("status", ""),
                                tasks=tasks_text
                            )
                            
                            llm_response = self.invoke_deepseek(prompt)
                            analysis_result = self.analysis_chains["jira"]["parser"].parse(llm_response)
                            
                            # Create result entry
                            result = {
                                "projectId": project_key,
                                "moduleId": epic.get("_id"),
                                "moduleTitle": epic.get("title"),
                                "featureId": feature.get("_id"),
                                "featureTitle": feature.get("title"),
                                "itemId": story.get("_id"),
                                "itemType": "story",
                                "itemTitle": story.get("title"),
                                "itemStatus": story.get("status", "unknown"),
                                "complianceItem": {
                                    "type": compliance_meta.get("type", ""),
                                    "id": article_id,
                                    "frameworkId": framework_id,
                                    "title": article_title,
                                    "content": item.page_content
                                },
                                "aligned": analysis_result.aligned,
                                "explanation": analysis_result.explanation,
                                "gaps": analysis_result.gaps,
                                "suggestions": analysis_result.suggestions,
                                "score": analysis_result.score,
                                "criticality": self._determine_criticality(
                                    analysis_result.score, 
                                    len(analysis_result.gaps)
                                ),
                                "jiraUpdates": analysis_result.jira_updates,
                                "analysisType": "jira",
                                "timestamp": datetime.now(),
                                "analyst": "AI-Analyzer"
                            }
                            
                            results.append(result)
                        
                        except Exception as e:
                            self.logger.error(f"Error analyzing story {story.get('_id')} against {article_id}: {e}")
        
        # Store results in MongoDB
        if results:
            self._store_gap_results(results)
        
        return {"project_key": project_key, "results_count": len(results)}
    
    def analyze_documents(self, project_key, documents=None):
        """Analyze documents against compliance templates"""
        self.logger.info(f"Starting document analysis for project {project_key}")
        
        # Get documents (either from parameter or fetch from Jira)
        if not documents:
            self.logger.info("No documents provided, fetching from Jira")
            documents = self._fetch_documents_from_jira(project_key)
        
        if not documents:
            self.logger.warning("No documents found for analysis")
            return {
                "project_key": project_key,
                "analysis_type": "document",
                "results_count": 0,
                "error": "No documents found for analysis"
            }
        
        # Process documents with LangChain
        results = []
        
        for doc_path in documents:
            try:
                # Process document
                doc_data = self.process_document(doc_path)
                
                # Find matching templates
                doc_content = doc_data["content"]
                matching_templates = self.vector_store.similarity_search(
                    doc_content,
                    k=1,
                    filter={"type": "template"}
                )
                
                if not matching_templates:
                    self.logger.warning(f"No matching template found for document {doc_path}")
                    continue
                
                # Get best matching template
                template = matching_templates[0]
                
                # Analyze document against template
                doc_sections_text = "\n".join([f"- {section['title']}" for section in doc_data["sections"]])
                
                prompt = self.analysis_chains["document"]["prompt"].format(
                    template_name=template.metadata.get("title", ""),
                    template_purpose="Compliance document template",
                    required_sections=template.page_content,
                    document_title=doc_data["metadata"].get("file_name", ""),
                    document_type=doc_data["metadata"].get("document_type", ""),
                    document_sections=doc_sections_text
                )
                
                llm_response = self.invoke_deepseek(prompt)
                analysis_result = self.analysis_chains["document"]["parser"].parse(llm_response)
                
                # Create result entry
                result = {
                    "projectId": project_key,
                    "moduleId": "documentation",
                    "moduleTitle": "Documentation",
                    "featureId": template.metadata.get("id", "template"),
                    "featureTitle": template.metadata.get("title", "Template"),
                    "itemId": doc_data["metadata"].get("file_name", ""),
                    "itemType": "document",
                    "itemTitle": doc_data["metadata"].get("file_name", ""),
                    "itemStatus": "existing",
                    "complianceItem": {
                        "type": "template",
                        "id": template.metadata.get("id", ""),
                        "frameworkId": "multiple",
                        "title": template.metadata.get("title", ""),
                        "content": template.page_content[:500]  # Truncate for MongoDB
                    },
                    "aligned": analysis_result.aligned,
                    "explanation": analysis_result.explanation,
                    "gaps": analysis_result.missing_sections + analysis_result.content_gaps,
                    "suggestions": analysis_result.improvement_suggestions,
                    "score": analysis_result.score,
                    "criticality": self._determine_criticality(
                        analysis_result.score, 
                        len(analysis_result.missing_sections) + len(analysis_result.content_gaps)
                    ),
                    "jiraUpdates": self._convert_to_jira_updates(analysis_result.tasks_to_create),
                    "analysisType": "document",
                    "timestamp": datetime.now(),
                    "analyst": "AI-Analyzer"
                }
                
                results.append(result)
            
            except Exception as e:
                self.logger.error(f"Error analyzing document {doc_path}: {e}")
        
        # Store results in MongoDB
        if results:
            self._store_gap_results(results)
        
        return {"project_key": project_key, "results_count": len(results)}
    
    def analyze_comprehensive(self, project_key, documents=None):
        """Perform comprehensive analysis using both Jira tasks and documents"""
        self.logger.info(f"Starting comprehensive analysis for project {project_key}")
        
        # Extract Jira data
        project_data = self.jira_connector.extract_project_data(project_key)
        
        # Get documents
        if not documents:
            documents = self._fetch_documents_from_jira(project_key)
        
        # Process documents
        processed_documents = []
        for doc_path in documents:
            try:
                doc_data = self.process_document(doc_path)
                processed_documents.append(doc_data)
            except Exception as e:
                self.logger.error(f"Error processing document {doc_path}: {e}")
        
        # Get all compliance requirements
        compliance_items = self.vector_store.similarity_search(
            "compliance requirement", 
            k=100,
            filter={"type": {"$in": ["article", "sub_article", "requirement"]}}
        )
        
        # Analyze against each compliance item
        results = []
        
        for compliance_item in compliance_items:
            try:
                # Extract compliance info
                compliance_meta = compliance_item.metadata
                framework_id = compliance_meta.get("frameworkId", "")
                article_id = compliance_meta.get("id", "")
                article_title = compliance_meta.get("title", "")
                
                # Find relevant tasks
                relevant_tasks = []
                for epic in project_data.get("epics", []):
                    for feature in epic.get("features", []):
                        for story in feature.get("stories", []):
                            # Simple relevance check - can be improved with embeddings
                            story_text = f"{story.get('title', '')}\n{story.get('storyFormat', '')}"
                            if (framework_id.lower() in story_text.lower() or 
                                article_id.lower() in story_text.lower()):
                                relevant_tasks.append({
                                    "epic": epic.get("title", ""),
                                    "feature": feature.get("title", ""),
                                    "title": story.get("title", ""),
                                    "description": story.get("description", ""),
                                    "user_story": story.get("storyFormat", ""),
                                    "status": story.get("status", "")
                                })
                
                # Find relevant documents
                relevant_docs = []
                for doc in processed_documents:
                    # Simple relevance check
                    if (framework_id.lower() in doc["content"].lower() or 
                        article_id.lower() in doc["content"].lower()):
                        relevant_docs.append({
                            "title": doc["metadata"].get("file_name", ""),
                            "type": doc["metadata"].get("document_type", ""),
                            "sections": [section["title"] for section in doc["sections"]]
                        })
                
                # Skip if no relevant tasks or documents
                if not relevant_tasks and not relevant_docs:
                    continue
                
                # Format data for analysis
                tasks_text = ""
                for task in relevant_tasks:
                    tasks_text += f"- {task['title']}: {task['description']}\n"
                    if task.get("user_story"):
                        tasks_text += f"  User Story: {task['user_story']}\n"
                    tasks_text += f"  Status: {task['status']}\n"
                    tasks_text += f"  Feature: {task['feature']}, Epic: {task['epic']}\n\n"
                
                docs_text = ""
                for doc in relevant_docs:
                    docs_text += f"- {doc['title']} ({doc['type']})\n"
                    for section in doc.get("sections", [])[:5]:  # Limit to first 5 sections
                        docs_text += f"  * {section}\n"
                    docs_text += "\n"
                
                # Run comprehensive analysis
                prompt = self.analysis_chains["comprehensive"]["prompt"].format(
                    framework_id=framework_id,
                    article_id=article_id,
                    article_title=article_title,
                    article_content=compliance_item.page_content,
                    tasks=tasks_text,
                    documents=docs_text
                )
                
                llm_response = self.invoke_deepseek(prompt)
                analysis_result = self.analysis_chains["comprehensive"]["parser"].parse(llm_response)
                
                # Create result entry
                result = {
                    "projectId": project_key,
                    "moduleId": self._map_compliance_to_module(compliance_meta),
                    "moduleTitle": self._get_module_title(compliance_meta),
                    "featureId": article_id,
                    "featureTitle": article_title,
                    "itemId": f"{framework_id}_{article_id}",
                    "itemType": "comprehensive",
                    "itemTitle": article_title,
                    "itemStatus": "analyzed",
                    "complianceItem": {
                        "type": compliance_meta.get("type", ""),
                        "id": article_id,
                        "frameworkId": framework_id,
                        "title": article_title,
                        "content": compliance_item.page_content[:500]  # Truncate for MongoDB
                    },
                    "aligned": analysis_result.aligned,
                    "explanation": analysis_result.explanation,
                    "gaps": analysis_result.task_gaps + analysis_result.documentation_gaps,
                    "suggestions": analysis_result.suggested_tasks + analysis_result.suggested_document_updates,
                    "score": analysis_result.score,
                    "criticality": self._determine_criticality(
                        analysis_result.score, 
                        len(analysis_result.task_gaps) + len(analysis_result.documentation_gaps)
                    ),
                    "jiraUpdates": self._convert_to_jira_updates(analysis_result.suggested_tasks),
                    "analysisType": "comprehensive",
                    "timestamp": datetime.now(),
                    "analyst": "AI-Analyzer"
                }
                
                results.append(result)
            
            except Exception as e:
                self.logger.error(f"Error in comprehensive analysis for {article_id}: {e}")
        
        # Store results in MongoDB
        if results:
            self._store_gap_results(results)
        
        return {"project_key": project_key, "results_count": len(results)}
    
    def _fetch_documents_from_jira(self, project_key):
        """Fetch document attachments from Jira"""
        try:
            # This would be your custom logic to get attachments from Jira
            # Here's a placeholder implementation
            attachments = self.jira_connector.get_project_attachments(project_key)
            
            # Download attachments to temp directory
            temp_dir = "temp_attachments"
            os.makedirs(temp_dir, exist_ok=True)
            
            document_paths = []
            for attachment in attachments:
                try:
                    file_path = os.path.join(temp_dir, attachment["filename"])
                    self.jira_connector.download_attachment(attachment["id"], file_path)
                    document_paths.append(file_path)
                except Exception as e:
                    self.logger.error(f"Error downloading attachment {attachment['filename']}: {e}")
            
            return document_paths
        
        except Exception as e:
            self.logger.error(f"Error fetching attachments from Jira: {e}")
            return []
    
    def _store_gap_results(self, results, analysis_type=None):
        """Store gap analysis results in MongoDB"""
        if not results:
            return
        
        for result in results:
            # Set analysis type if provided
            if analysis_type:
                result["analysisType"] = analysis_type
            
            # Store in MongoDB
            self.db.gapAnalysisResults.replace_one(
                {
                    "projectId": result["projectId"],
                    "itemId": result["itemId"],
                    "complianceItem.id": result["complianceItem"]["id"],
                    "analysisType": result["analysisType"]
                },
                result,
                upsert=True
            )
    
    def _map_compliance_to_module(self, compliance_meta):
        """Map compliance item to a module"""
        # Simple mapping based on framework and article ID
        framework_id = compliance_meta.get("frameworkId", "")
        
        # This would be your custom mapping logic
        # Here's a simplified placeholder version
        if framework_id == "euAiAct":
            article_id = compliance_meta.get("id", "")
            if "6" in article_id or "9" in article_id:
                return "risk-management"
            elif "10" in article_id:
                return "data-governance"
            elif "11" in article_id or "12" in article_id:
                return "documentation"
            elif "13" in article_id or "14" in article_id:
                return "human-ai"
            else:
                return "other"
        else:
            return "other"
    
    def _get_module_title(self, compliance_meta):
        """Get module title from compliance metadata"""
        module_id = self._map_compliance_to_module(compliance_meta)
        
        # Map module ID to title
        module_titles = {
            "risk-management": "Risk Management and Classification",
            "data-governance": "Data Governance and Management",
            "documentation": "Technical Documentation and Record-Keeping",
            "system-development": "AI System Development and Performance",
            "human-ai": "Human-AI Interaction and Transparency",
            "other": "Other Requirements"
        }
        
        return module_titles.get(module_id, "Other Requirements")
    
    def _determine_criticality(self, score, gaps_count):
        """Determine criticality based on score and number of gaps"""
        if score < 50 or gaps_count > 3:
            return "high"
        elif score < 80 or gaps_count > 0:
            return "medium"
        else:
            return "low"
    
    def _convert_to_jira_updates(self, suggested_tasks):
        """Convert suggested tasks to Jira update format"""
        jira_updates = []
        
        for task in suggested_tasks:
            jira_updates.append({
                "type": "new_issue",
                "target": "",  # Will be filled by Jira updater
                "content": task
            })
        
        return jira_updates
