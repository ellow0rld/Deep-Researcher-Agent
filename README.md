# Deep-Researcher-Agent
A local AI-powered research assistant that helps you analyze documents, answer complex queries, and generate detailed research reports. Built with Python, Streamlit, and local embedding models, it provides multi-step reasoning, semantic search, and exportable reports.

## Features
1. Document Upload: Upload TXT, PDF, or JSON files to build a local knowledge base.
2. Semantic Search: Uses local embeddings (all-MiniLM-L6-v2) to find the most relevant documents for your queries.
3. Multi-Step Reasoning: Breaks down complex queries into subtasks and provides structured answers.
4. Full-Content Responses: Returns detailed answers for each subtask using the full content of the documents.
5. Conversation History: Maintains chat history across queries.
6. Exportable Reports: Download your session as PDF or Markdown with answers and document analysis.
7. Analysis Panel: Optional view of contributing documents and similarity scores.

## How It Works

1. Document Ingestion: The agent processes uploaded documents and stores them with embeddings for semantic search.
2. Query Processing: User queries are split into subtasks using MultiStepReasoner.
3. Answer Generation: For each subtask, the agent retrieves relevant documents and generates full-content answers.
4. Analysis: Each response can show contributing documents and similarity scores.
5. Export: Complete session reports can be exported as PDF or Markdown.

## Future Improvements
1. Integrate LLM for more natural answers.
2. Support additional file formats (DOCX, HTML, etc.).
3. Add advanced filtering or weighting for semantic search.
4. Allow inline highlighting of document content in answers.

## Web URL
https://deep-researcher-agent-1.streamlit.app/

# Screenshots
## Home Page
<img width="2275" height="1488" alt="image" src="https://github.com/user-attachments/assets/4d43202f-467b-49d1-ad53-acdf76a2a7fd" />
## Upload Documents
<img width="1885" height="836" alt="image" src="https://github.com/user-attachments/assets/f418d7b2-754d-47eb-a89a-30085046820b" />
## Query
<img width="1504" height="1383" alt="image" src="https://github.com/user-attachments/assets/f49d2a4a-f448-4549-b581-9023e6d22d24" />
## Follow ups/New query
<img width="1672" height="1612" alt="image" src="https://github.com/user-attachments/assets/9024ccf0-9c68-484b-b19e-5905b94deff4" />
