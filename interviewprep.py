import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import json
import re
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
import time

# Page configuration
st.set_page_config(
    page_title="Interview Prep Master",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #667eea;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .question-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .question-text {
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .answer-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .answer-text {
        color: #2d3748;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .video-link {
        background: #48bb78;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        text-decoration: none;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    .stButton>button {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Mock data generator (simulates web scraping)
def generate_mock_questions(technology, num_questions, company=None):
    """Generate mock interview questions and answers"""
    
    base_questions = {
        "AI": [
            {
                "question": "What is the difference between supervised and unsupervised learning?",
                "answer": "Supervised learning uses labeled data where the target outcome is known, like classification or regression. Unsupervised learning works with unlabeled data to find patterns, like clustering or dimensionality reduction. Semi-supervised learning combines both approaches.",
                "video": "https://www.youtube.com/watch?v=rHeaoaiBM6Y"
            },
            {
                "question": "Explain the concept of overfitting and how to prevent it.",
                "answer": "Overfitting occurs when a model learns training data too well, including noise, leading to poor generalization. Prevention methods include: cross-validation, regularization (L1/L2), dropout, early stopping, data augmentation, and using simpler models.",
                "video": "https://www.youtube.com/watch?v=DEMmkFC6IGM"
            },
            {
                "question": "What is gradient descent and its variants?",
                "answer": "Gradient descent is an optimization algorithm that iteratively adjusts parameters to minimize loss. Variants include: Batch GD (uses all data), Stochastic GD (uses one sample), Mini-batch GD (uses small batches), Adam, RMSprop, and Adagrad optimizers.",
                "video": "https://www.youtube.com/watch?v=sDv4f4s2SB8"
            },
            {
                "question": "What are transformers and attention mechanisms?",
                "answer": "Transformers are neural network architectures using self-attention mechanisms to process sequential data. Attention allows models to focus on relevant parts of input. They power models like BERT, GPT, and enable parallel processing unlike RNNs.",
                "video": "https://www.youtube.com/watch?v=SZorAJ4I-sA"
            },
            {
                "question": "Explain bias-variance tradeoff in machine learning.",
                "answer": "Bias is error from overly simplistic assumptions (underfitting), variance is error from sensitivity to training data (overfitting). The tradeoff is finding the balance - reducing one often increases the other. Optimal model complexity minimizes total error.",
                "video": "https://www.youtube.com/watch?v=EuBBz3bI-aA"
            }
        ],
        "ML": [
            {
                "question": "What are ensemble methods and their types?",
                "answer": "Ensemble methods combine multiple models for better predictions. Types include: Bagging (Random Forest, reduces variance), Boosting (XGBoost, AdaBoost, reduces bias), Stacking (meta-learning), and Voting (majority/average predictions).",
                "video": "https://www.youtube.com/watch?v=Un9zObFjBH0"
            },
            {
                "question": "Explain feature engineering and its importance.",
                "answer": "Feature engineering creates new features from existing data to improve model performance. Techniques include: scaling, encoding categorical variables, creating interaction terms, polynomial features, binning, and domain-specific transformations. It often impacts performance more than algorithm choice.",
                "video": "https://www.youtube.com/watch?v=6WDFfaYtN6s"
            },
            {
                "question": "What is cross-validation and why is it important?",
                "answer": "Cross-validation assesses model performance on unseen data by splitting data into folds. K-fold CV uses K subsets, training on K-1 and validating on 1 repeatedly. It provides robust performance estimates, helps detect overfitting, and guides hyperparameter tuning.",
                "video": "https://www.youtube.com/watch?v=fSytzGwwBVw"
            },
            {
                "question": "Describe dimensionality reduction techniques.",
                "answer": "Techniques reduce feature count while preserving information. PCA (linear transformation), t-SNE (visualization), UMAP (manifold learning), LDA (supervised), and feature selection methods. Benefits include faster training, reduced overfitting, and easier visualization.",
                "video": "https://www.youtube.com/watch?v=FgakZw6K1QQ"
            },
            {
                "question": "What is the curse of dimensionality?",
                "answer": "In high-dimensional spaces, data becomes sparse and distance metrics lose meaning. More features need exponentially more data. It causes overfitting, increased computation, and degraded model performance. Addressed through dimensionality reduction and feature selection.",
                "video": "https://www.youtube.com/watch?v=QZ0DtNFdDko"
            }
        ],
        "Python": [
            {
                "question": "Explain Python decorators and their use cases.",
                "answer": "Decorators modify function behavior without changing code. Syntax: @decorator above function. Common uses: logging, timing, authentication, caching (@lru_cache), validation. They enable clean, reusable code for cross-cutting concerns.",
                "video": "https://www.youtube.com/watch?v=FsAPt_9Bf3U"
            },
            {
                "question": "What are generators and their advantages?",
                "answer": "Generators are functions using 'yield' to return iterators lazily. They produce values on-demand, saving memory for large datasets. Use cases: processing large files, infinite sequences, pipeline processing. More memory-efficient than lists.",
                "video": "https://www.youtube.com/watch?v=bD05uGo_sVI"
            },
            {
                "question": "Explain list comprehensions vs generator expressions.",
                "answer": "List comprehensions [x for x in range(10)] create lists immediately in memory. Generator expressions (x for x in range(10)) create iterators lazily. Generators use less memory, ideal for large data or one-time iteration.",
                "video": "https://www.youtube.com/watch?v=5jwV3zxXc8E"
            },
            {
                "question": "What is the GIL and its impact on multithreading?",
                "answer": "Global Interpreter Lock (GIL) allows only one thread to execute Python bytecode at a time. It limits true parallelism for CPU-bound tasks. Solutions: multiprocessing for CPU-bound, threading for I/O-bound, or alternative implementations (Jython, IronPython).",
                "video": "https://www.youtube.com/watch?v=Obt-vMVdM8s"
            },
            {
                "question": "Explain context managers and the 'with' statement.",
                "answer": "Context managers manage resources (files, connections) automatically. The 'with' statement ensures cleanup via __enter__ and __exit__ methods. Example: 'with open(file) as f:' automatically closes file. Create custom ones using @contextmanager decorator.",
                "video": "https://www.youtube.com/watch?v=-aKFBoZpiqA"
            }
        ],
        "Azure Cloud": [
            {
                "question": "Explain Azure Service Bus vs Event Grid vs Event Hub.",
                "answer": "Service Bus: Enterprise messaging, queues/topics, guaranteed delivery. Event Grid: Event-driven reactive programming, pub-sub model. Event Hub: Big data streaming, millions of events/sec. Choose based on messaging pattern, throughput, and delivery guarantees.",
                "video": "https://www.youtube.com/watch?v=jfKv2u7KM9k"
            },
            {
                "question": "What is Azure Functions and its use cases?",
                "answer": "Serverless compute service for event-driven code execution. Use cases: API backends, scheduled tasks, data processing, IoT processing. Supports multiple languages, auto-scaling, pay-per-execution. Integrates with bindings for easy service connection.",
                "video": "https://www.youtube.com/watch?v=8-jz5f_JyEQ"
            },
            {
                "question": "Describe Azure Virtual Network and network security.",
                "answer": "VNet provides isolated network in Azure. Features: subnets, NSGs (Network Security Groups), route tables, VPN Gateway, ExpressRoute. Security: service endpoints, private endpoints, firewall rules, DDoS protection. Enables hybrid cloud connectivity.",
                "video": "https://www.youtube.com/watch?v=MJL7dSJRW8w"
            },
            {
                "question": "What is Azure Kubernetes Service (AKS)?",
                "answer": "Managed Kubernetes service for containerized applications. Features: automatic upgrades, scaling, monitoring, Azure AD integration. Benefits: simplified cluster management, built-in security, CI/CD integration, cost optimization with node auto-scaling.",
                "video": "https://www.youtube.com/watch?v=4ht22ReBjno"
            },
            {
                "question": "Explain Azure Storage types and their use cases.",
                "answer": "Blob Storage: unstructured data (images, videos). File Storage: SMB file shares. Queue Storage: message queuing. Table Storage: NoSQL key-value. Disk Storage: VM disks. Each has different performance tiers (hot, cool, archive) and redundancy options.",
                "video": "https://www.youtube.com/watch?v=9S7OiZU_f7M"
            }
        ],
        "AWS": [
            {
                "question": "Explain AWS Lambda and its best practices.",
                "answer": "Serverless compute service for event-driven functions. Best practices: minimize cold starts, use environment variables, implement proper error handling, optimize memory allocation, leverage layers for dependencies, use VPC when needed, implement proper logging with CloudWatch.",
                "video": "https://www.youtube.com/watch?v=eOBq__h4OJ4"
            },
            {
                "question": "What is AWS IAM and security best practices?",
                "answer": "Identity and Access Management controls AWS resource access. Best practices: use roles over access keys, apply least privilege, enable MFA, rotate credentials, use IAM policies and SCPs, implement resource-based policies, audit with CloudTrail.",
                "video": "https://www.youtube.com/watch?v=iF9fs8Rw4Uo"
            },
            {
                "question": "Describe S3 storage classes and lifecycle policies.",
                "answer": "Storage classes: Standard (frequent access), Intelligent-Tiering (auto-optimization), Standard-IA (infrequent), One Zone-IA, Glacier (archive), Glacier Deep Archive. Lifecycle policies automate transitions between classes and expiration to optimize costs.",
                "video": "https://www.youtube.com/watch?v=rHeTn9pHNKo"
            },
            {
                "question": "What is Amazon ECS vs EKS?",
                "answer": "ECS (Elastic Container Service): AWS-native container orchestration, simpler, tighter AWS integration. EKS (Elastic Kubernetes Service): managed Kubernetes, more portable, larger ecosystem. Choose ECS for AWS-centric, EKS for multi-cloud or K8s expertise.",
                "video": "https://www.youtube.com/watch?v=AYAh6YDXuho"
            },
            {
                "question": "Explain AWS VPC and networking components.",
                "answer": "VPC (Virtual Private Cloud) isolates AWS resources. Components: subnets (public/private), Internet Gateway, NAT Gateway, route tables, security groups, NACLs, VPC peering, Transit Gateway. Enables secure, scalable network architecture with fine-grained control.",
                "video": "https://www.youtube.com/watch?v=bGDMeD6kOz0"
            }
        ],
        "GCP Cloud": [
            {
                "question": "What is Google Cloud Functions and use cases?",
                "answer": "Serverless execution environment for event-driven code. Triggers: HTTP, Pub/Sub, Cloud Storage, Firestore. Use cases: webhooks, data processing, API backends, IoT. Supports multiple languages, auto-scaling, integrated with GCP services. Pay only for execution time.",
                "video": "https://www.youtube.com/watch?v=1r3vMYywNLo"
            },
            {
                "question": "Explain BigQuery and its optimization techniques.",
                "answer": "Serverless data warehouse for analytics. Optimization: partition tables (date/time), cluster columns (filter/group columns), use denormalized data, avoid SELECT *, materialize frequently queried data, use streaming inserts wisely, leverage BI Engine for caching.",
                "video": "https://www.youtube.com/watch?v=d3MDxC_iuaw"
            },
            {
                "question": "What is Google Kubernetes Engine (GKE)?",
                "answer": "Managed Kubernetes service with autopilot mode. Features: auto-upgrade, auto-repair, built-in monitoring with Cloud Operations, Workload Identity for security, Binary Authorization, GKE Autopilot (fully managed). Integrates with Cloud Build for CI/CD.",
                "video": "https://www.youtube.com/watch?v=8tg_WyhN2ps"
            },
            {
                "question": "Describe Cloud Storage classes and use cases.",
                "answer": "Standard: frequent access, low latency. Nearline: accessed <once/month. Coldline: accessed <once/quarter. Archive: accessed <once/year. Auto-class automatically transitions objects. Use lifecycle policies for automatic management and cost optimization.",
                "video": "https://www.youtube.com/watch?v=h8GH5X_q3-Y"
            },
            {
                "question": "What is Cloud Pub/Sub and its patterns?",
                "answer": "Fully managed messaging service for event-driven systems. Patterns: fan-out (broadcast), fan-in (aggregate), load balancing, streaming. Features: at-least-once delivery, ordering, dead letter topics, message filtering. Integrates with Dataflow for stream processing.",
                "video": "https://www.youtube.com/watch?v=cvu53CnZmGI"
            }
        ],
        "MLOps": [
            {
                "question": "What is MLOps and its core principles?",
                "answer": "MLOps applies DevOps to ML lifecycle. Principles: continuous integration (code/data/model), automated testing, versioning (data, code, models), monitoring, reproducibility, CI/CD pipelines. Tools: MLflow, Kubeflow, Azure ML, SageMaker. Bridges ML and operations.",
                "video": "https://www.youtube.com/watch?v=ZVWg18AXXuE"
            },
            {
                "question": "Explain model versioning and experiment tracking.",
                "answer": "Version control for models, data, and parameters. Tools: MLflow, DVC, Weights & Biases. Track metrics, hyperparameters, artifacts. Benefits: reproducibility, compare experiments, rollback models, audit trail. Essential for production ML systems.",
                "video": "https://www.youtube.com/watch?v=KNslAw0iV0w"
            },
            {
                "question": "What is feature store and its benefits?",
                "answer": "Centralized repository for ML features. Benefits: feature reusability, consistency between training/serving, reduce data redundancy, versioning, lineage tracking. Tools: Feast, Tecton, Databricks Feature Store. Enables feature discovery and governance.",
                "video": "https://www.youtube.com/watch?v=PtoWbAKGcS0"
            },
            {
                "question": "Describe model monitoring and drift detection.",
                "answer": "Monitor model performance in production. Detect data drift (input distribution change), concept drift (target relationship change), prediction drift. Metrics: accuracy degradation, latency, data quality. Tools: Evidently AI, Fiddler, Arize. Trigger retraining when drift detected.",
                "video": "https://www.youtube.com/watch?v=8x9UbFEeWYU"
            },
            {
                "question": "What are ML pipelines and orchestration?",
                "answer": "Automated workflows for ML tasks: data ingestion, preprocessing, training, validation, deployment. Orchestration tools: Kubeflow Pipelines, Airflow, Prefect, Azure ML Pipelines. Benefits: automation, reproducibility, scalability, scheduling. DAG-based execution.",
                "video": "https://www.youtube.com/watch?v=x0G4WQPB2Ds"
            }
        ],
        "AI Engineer": [
            {
                "question": "How do you design a scalable ML system architecture?",
                "answer": "Components: data pipeline (batch/streaming), feature store, model training (distributed), model registry, serving layer (REST/gRPC), monitoring. Consider: latency requirements, throughput, cost, fault tolerance. Use microservices, containerization, auto-scaling.",
                "video": "https://www.youtube.com/watch?v=QEOQvvhTuyU"
            },
            {
                "question": "Explain model serving strategies: batch vs real-time.",
                "answer": "Batch: periodic predictions, high throughput, latency-tolerant. Real-time: low latency (<100ms), on-demand predictions. Hybrid: pre-compute common cases, real-time for rest. Tools: TF Serving, Seldon, KServe, SageMaker. Consider caching, load balancing.",
                "video": "https://www.youtube.com/watch?v=0aeMRPQ2q8Y"
            },
            {
                "question": "What is A/B testing for ML models?",
                "answer": "Compare model performance with control/variant in production. Metrics: statistical significance, business KPIs, user experience. Implement: traffic splitting, metric collection, analysis. Tools: Optimizely, LaunchDarkly. Enables data-driven model selection and gradual rollouts.",
                "video": "https://www.youtube.com/watch?v=8u7GRkMxIkg"
            },
            {
                "question": "How do you handle model reproducibility?",
                "answer": "Version control: code (git), data (DVC), models (registry), environment (Docker). Set random seeds, document dependencies, log experiments, use configuration files. Tools: MLflow, DVC, Neptune.ai. Enables debugging, compliance, and collaboration.",
                "video": "https://www.youtube.com/watch?v=TWxwfMDFJYg"
            },
            {
                "question": "Explain the concept of model explainability.",
                "answer": "Understand model decisions for trust and debugging. Techniques: SHAP (Shapley values), LIME (local approximations), attention visualization, feature importance. Benefits: debug errors, regulatory compliance, user trust. Critical for high-stakes applications.",
                "video": "https://www.youtube.com/watch?v=VB9uV-x0gtg"
            }
        ],
        "ETL": [
            {
                "question": "What is ETL vs ELT and when to use each?",
                "answer": "ETL (Extract-Transform-Load): transform before loading, traditional warehouses. ELT (Extract-Load-Transform): load raw data then transform, modern cloud warehouses. Use ETL for complex transformations, legacy systems. Use ELT for cloud, big data, flexibility.",
                "video": "https://www.youtube.com/watch?v=OW5OgsLpDCQ"
            },
            {
                "question": "Explain data pipeline orchestration tools.",
                "answer": "Apache Airflow: Python-based DAGs, extensive integrations. Prefect: modern Python, dynamic workflows. Luigi: Spotify's tool, simpler. Azure Data Factory: cloud-native, visual. AWS Glue: serverless, integrated. Choose based on ecosystem, complexity, team skills.",
                "video": "https://www.youtube.com/watch?v=AHMm1wfGuHE"
            },
            {
                "question": "What are slowly changing dimensions (SCD)?",
                "answer": "Track historical changes in dimension tables. Type 1: overwrite (no history). Type 2: add new row (full history). Type 3: add column (limited history). Type 4: separate history table. Type 6: hybrid. Choose based on tracking needs and storage.",
                "video": "https://www.youtube.com/watch?v=c4kE5BriO5w"
            },
            {
                "question": "How do you handle incremental data loads?",
                "answer": "Strategies: timestamp-based (last_modified), change data capture (CDC), log-based replication, trigger-based. Handle: new records (INSERT), updates (UPSERT/MERGE), deletes (soft delete flags). Tools: Debezium, Fivetran, AWS DMS. Optimize with partitioning.",
                "video": "https://www.youtube.com/watch?v=4gFZ7PgvHDw"
            },
            {
                "question": "What is data quality and validation in ETL?",
                "answer": "Ensure data accuracy, completeness, consistency. Checks: null values, data types, ranges, formats, referential integrity, duplicates. Implement: data profiling, validation rules, quality metrics, alerts. Tools: Great Expectations, Deequ, dbt tests. Critical for trustworthy analytics.",
                "video": "https://www.youtube.com/watch?v=T5NQ_pBCHRw"
            }
        ],
        "Data Engineer": [
            {
                "question": "Explain data lake vs data warehouse architecture.",
                "answer": "Data Lake: raw data, all formats, schema-on-read, cost-effective, exploratory. Data Warehouse: structured, schema-on-write, optimized queries, business analytics. Modern: data lakehouse (combines both). Tools: S3+Athena (lake), Snowflake (warehouse), Databricks (lakehouse).",
                "video": "https://www.youtube.com/watch?v=5V9W9l3z-0I"
            },
            {
                "question": "What is Apache Spark and its components?",
                "answer": "Distributed processing framework. Components: Spark Core (RDDs), Spark SQL (DataFrames), Streaming (real-time), MLlib (ML), GraphX (graphs). Features: in-memory processing, lazy evaluation, DAG optimization. Use for: large-scale ETL, ML, streaming.",
                "video": "https://www.youtube.com/watch?v=Q5G8PaNQ1mo"
            },
            {
                "question": "Describe partitioning and bucketing strategies.",
                "answer": "Partitioning: organize data by column values (date, region), reduces scan. Bucketing: distribute data across fixed buckets, optimizes joins. Hive-style partitioning for storage. Use for query performance, parallel processing. Consider cardinality and query patterns.",
                "video": "https://www.youtube.com/watch?v=2wY7RP7TFC8"
            },
            {
                "question": "What is Change Data Capture (CDC)?",
                "answer": "Identify and capture database changes in real-time. Methods: log-based (transaction logs), trigger-based, timestamp-based, snapshot comparison. Tools: Debezium, AWS DMS, Oracle GoldenGate. Use cases: replication, real-time analytics, event sourcing, audit trails.",
                "video": "https://www.youtube.com/watch?v=aYMdEe-vQnw"
            },
            {
                "question": "Explain streaming vs batch processing.",
                "answer": "Batch: process large volumes periodically, high throughput, eventual consistency. Streaming: continuous processing, low latency, real-time insights. Tools: Kafka, Flink, Spark Streaming, Kinesis. Lambda architecture (both), Kappa (streaming only). Choose based on latency, volume, use case.",
                "video": "https://www.youtube.com/watch?v=YJ0CAbDWq_A"
            }
        ],
        "DevOps": [
            {
                "question": "Explain CI/CD pipeline best practices.",
                "answer": "Continuous Integration: automated builds, tests on commits, fast feedback. Continuous Delivery: automated deployment to staging. Continuous Deployment: automated production release. Best practices: version control, automated tests, infrastructure as code, blue-green deployments, rollback capability.",
                "video": "https://www.youtube.com/watch?v=scEDHsr3APg"
            },
            {
                "question": "What is Infrastructure as Code (IaC)?",
                "answer": "Manage infrastructure through code. Tools: Terraform (multi-cloud), CloudFormation (AWS), ARM templates (Azure), Pulumi (programming languages). Benefits: version control, reproducibility, consistency, automation. Best practices: modules, state management, testing, documentation.",
                "video": "https://www.youtube.com/watch?v=POPP2WTJ8es"
            },
            {
                "question": "Describe container orchestration with Kubernetes.",
                "answer": "Kubernetes automates deployment, scaling, management of containers. Components: Pods, Services, Deployments, ConfigMaps, Secrets, Ingress. Features: self-healing, auto-scaling, rolling updates, service discovery. Benefits: portability, high availability, resource efficiency.",
                "video": "https://www.youtube.com/watch?v=X48VuDVv0do"
            },
            {
                "question": "What is monitoring and observability?",
                "answer": "Monitoring: metrics, logs, alerts for known issues. Observability: understand system behavior, debug unknowns. Three pillars: logs (events), metrics (numbers), traces (requests). Tools: Prometheus, Grafana, ELK Stack, Datadog, New Relic. Implement SLIs, SLOs, error budgets.",
                "video": "https://www.youtube.com/watch?v=0SdE9zpCHrs"
            },
            {
                "question": "Explain blue-green and canary deployments.",
                "answer": "Blue-Green: two identical environments, instant switch, easy rollback. Canary: gradual rollout to subset of users, monitor metrics, progressive traffic shift. Benefits: reduced risk, zero downtime, easy rollback. Implement with load balancers, feature flags, service mesh.",
                "video": "https://www.youtube.com/watch?v=7hlIqKElhFo"
            }
        ]
    }
    
    # Get questions for the technology or use AI questions as default
    questions = base_questions.get(technology, base_questions["AI"])
    
    # Adjust for company-specific context if provided
    if company and company != "Select Company":
        for q in questions:
            q["question"] = f"[{company}] {q['question']}"
    
    # Return requested number of questions
    return questions[:min(num_questions, len(questions))]


def create_pdf(questions_data, technology, company=None):
    """Create PDF document"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=RGBColor(102, 126, 234),
        spaceAfter=30,
        alignment=TA_LEFT
    )
    
    question_style = ParagraphStyle(
        'Question',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=RGBColor(118, 75, 162),
        spaceAfter=10,
        spaceBefore=15
    )
    
    answer_style = ParagraphStyle(
        'Answer',
        parent=styles['BodyText'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=10
    )
    
    content = []
    
    # Title
    title_text = f"Interview Questions: {technology}"
    if company and company != "Select Company":
        title_text += f" - {company}"
    
    content.append(Paragraph(title_text, title_style))
    content.append(Spacer(1, 0.2*inch))
    
    # Add date
    date_text = f"Generated on: {datetime.now().strftime('%B %d, %Y')}"
    content.append(Paragraph(date_text, styles['Normal']))
    content.append(Spacer(1, 0.3*inch))
    
    # Add questions and answers
    for idx, qa in enumerate(questions_data, 1):
        # Question
        q_text = f"Q{idx}. {qa['question']}"
        content.append(Paragraph(q_text, question_style))
        
        # Answer
        a_text = f"<b>Answer:</b> {qa['answer']}"
        content.append(Paragraph(a_text, answer_style))
        
        # Video link
        if qa.get('video'):
            v_text = f"<b>Video Resource:</b> <link href='{qa['video']}'>{qa['video']}</link>"
            content.append(Paragraph(v_text, styles['Normal']))
        
        content.append(Spacer(1, 0.2*inch))
    
    doc.build(content)
    buffer.seek(0)
    return buffer


def create_word(questions_data, technology, company=None):
    """Create Word document"""
    doc = Document()
    
    # Title
    title = doc.add_heading('Interview Questions & Answers', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Subtitle
    subtitle_text = f"{technology}"
    if company and company != "Select Company":
        subtitle_text += f" - {company} Interview Preparation"
    subtitle = doc.add_heading(subtitle_text, level=2)
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Date
    date_para = doc.add_paragraph()
    date_run = date_para.add_run(f"Generated on: {datetime.now().strftime('%B %d, %Y')}")
    date_run.font.size = Pt(10)
    date_run.font.color.rgb = RGBColor(128, 128, 128)
    date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_paragraph()
    
    # Add questions and answers
    for idx, qa in enumerate(questions_data, 1):
        # Question
        q_heading = doc.add_heading(f"Question {idx}", level=2)
        q_heading.runs[0].font.color.rgb = RGBColor(102, 126, 234)
        
        q_para = doc.add_paragraph(qa['question'])
        q_para.runs[0].font.bold = True
        q_para.runs[0].font.size = Pt(12)
        
        # Answer
        a_heading = doc.add_paragraph()
        a_run = a_heading.add_run("Answer:")
        a_run.font.bold = True
        a_run.font.color.rgb = RGBColor(118, 75, 162)
        a_run.font.size = Pt(11)
        
        answer_para = doc.add_paragraph(qa['answer'])
        answer_para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        
        # Video link
        if qa.get('video'):
            v_para = doc.add_paragraph()
            v_run = v_para.add_run("Video Resource: ")
            v_run.font.bold = True
            v_link = v_para.add_run(qa['video'])
            v_link.font.color.rgb = RGBColor(72, 187, 120)
            
        doc.add_paragraph()
    
    # Save to buffer
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer


# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 Interview Prep Master</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Your Ultimate Guide to Technical Interview Success</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Technology selection
        tech_options = [
            "AI", "ML", "Python", "Azure Cloud", "AWS", "GCP Cloud",
            "MLOps", "AI Engineer", "ETL", "Data Engineer", "DevOps"
        ]
        
        selected_tech = st.selectbox(
            "🔧 Select Technology",
            tech_options,
            index=0
        )
        
        # Custom technology input
        use_custom = st.checkbox("Use Custom Technology")
        if use_custom:
            custom_tech = st.text_input("Enter Custom Technology", placeholder="e.g., React, Kubernetes")
            if custom_tech:
                selected_tech = custom_tech
        
        st.markdown("---")
        
        # Number of questions
        num_questions = st.select_slider(
            "📊 Number of Questions",
            options=[5, 10, 20, 30, 40, 50],
            value=10
        )
        
        st.markdown("---")
        
        # Company selection
        st.markdown("### 🏢 Company-Specific Prep")
        company_options = ["Select Company", "Infosys", "TCS", "Wipro", "Accenture", "Cognizant", 
                          "Google", "Amazon", "Microsoft", "Meta", "Apple"]
        
        selected_company = st.selectbox("Select Company", company_options)
        
        # Custom company
        use_custom_company = st.checkbox("Use Custom Company")
        if use_custom_company:
            custom_company = st.text_input("Enter Company Name", placeholder="e.g., Deloitte")
            if custom_company:
                selected_company = custom_company
        
        st.markdown("---")
        
        # Info section
        st.markdown("### 💡 Tips")
        st.info("💎 Practice consistently\n\n🎥 Watch video explanations\n\n📝 Take notes\n\n🔄 Review regularly")
    
    # Main content area
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card"><h3>🎯</h3><p>Technology</p><h4>' + selected_tech + '</h4></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card"><h3>📚</h3><p>Questions</p><h4>' + str(num_questions) + '</h4></div>', unsafe_allow_html=True)
    
    with col3:
        company_display = selected_company if selected_company != "Select Company" else "General"
        st.markdown('<div class="metric-card"><h3>🏢</h3><p>Company</p><h4>' + company_display + '</h4></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Generate button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Generate Interview Questions", use_container_width=True):
            with st.spinner("🔍 Fetching latest interview questions..."):
                time.sleep(1)  # Simulate API call
                
                # Generate questions
                company_for_gen = selected_company if selected_company != "Select Company" else None
                questions_data = generate_mock_questions(selected_tech, num_questions, company_for_gen)
                
                # Store in session state
                st.session_state['questions_data'] = questions_data
                st.session_state['tech'] = selected_tech
                st.session_state['company'] = selected_company
                
                st.success(f"✅ Successfully generated {len(questions_data)} questions!")
    
    # Display questions
    if 'questions_data' in st.session_state:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">📋 Interview Questions & Answers</h2>', unsafe_allow_html=True)
        
        questions_data = st.session_state['questions_data']
        
        # Display each question
        for idx, qa in enumerate(questions_data, 1):
            with st.container():
                # Question card
                st.markdown(f"""
                <div class="question-card">
                    <div class="question-text">Q{idx}. {qa['question']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Answer card
                st.markdown(f"""
                <div class="answer-card">
                    <div class="answer-text">
                        <strong>Answer:</strong><br>
                        {qa['answer']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Video link
                if qa.get('video'):
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.markdown(f"""
                        <a href="{qa['video']}" target="_blank" class="video-link">
                            🎥 Watch Video Explanation
                        </a>
                        """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
        
        # Export options
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">💾 Export Options</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown("<br>", unsafe_allow_html=True)
        
        with col2:
            # PDF download
            pdf_buffer = create_pdf(
                questions_data, 
                st.session_state['tech'],
                st.session_state['company'] if st.session_state['company'] != "Select Company" else None
            )
            
            filename_base = f"{st.session_state['tech']}_Interview_Questions"
            if st.session_state['company'] != "Select Company":
                filename_base += f"_{st.session_state['company']}"
            
            st.download_button(
                label="📄 Download as PDF",
                data=pdf_buffer,
                file_name=f"{filename_base}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        
        with col3:
            # Word download
            word_buffer = create_word(
                questions_data,
                st.session_state['tech'],
                st.session_state['company'] if st.session_state['company'] != "Select Company" else None
            )
            
            st.download_button(
                label="📝 Download as Word",
                data=word_buffer,
                file_name=f"{filename_base}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🌟 <strong>Interview Prep Master</strong> - Powered by AI & Real Interview Experiences</p>
        <p>💼 Practice • 🎯 Prepare • 🚀 Succeed</p>
        <p style="font-size: 0.9rem;">Note: This app uses curated content and simulates web scraping for demonstration purposes.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
