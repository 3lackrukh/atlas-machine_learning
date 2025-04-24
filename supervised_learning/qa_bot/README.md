# QA Bot 🤖💬

A Question-Answering system that leverages BERT and semantic search to answer questions from reference documents.

## 📋 Project Overview

This project implements a question-answering system that can find relevant information in a corpus of documents and provide concise answers to natural language questions. The system uses two main techniques:

1. **Semantic Search** 🔍: Finding the most relevant document to a given question using sentence embeddings
2. **Question Answering** ❓: Extracting the specific snippet of text that answers the question using BERT

## 🛠️ Installation

### Prerequisites

- Python 3.9
- Ubuntu 20.04 LTS (recommended)

### Dependencies

Install the required dependencies:

```bash
# Install TensorFlow Hub
pip install --user tensorflow-hub==0.15.0

# Install Transformers
pip install --user transformers==4.44.2

# Install TensorFlow (if not already installed)
pip install --user tensorflow==2.15

# Install NumPy (if not already installed)
pip install --user numpy==1.25.2
```

### Dataset

The project uses a collection of Holberton USA Zendesk Articles. You should extract the `ZendeskArticles.zip` file in the project directory.

## 📁 Project Structure

- **0-qa.py**: Core question-answering function that uses BERT to extract answers from reference text
- **1-loop.py**: Simple interactive loop that takes user questions and exits on specific commands
- **2-qa.py**: Combines the QA function with an interactive loop to answer questions from a reference document
- **3-semantic_search.py**: Implements semantic search to find the most relevant document to a question
- **4-qa.py**: Complete QA system that uses semantic search to find relevant documents and extracts answers

### Main Files and Their Purpose

| File | Description |
|------|-------------|
| `0-qa.py` | Provides the core `question_answer` function that uses BERT to find answers in a reference text |
| `1-loop.py` | Simple interactive prompt that accepts user input and responds until exit commands are given |
| `2-qa.py` | Combines question answering with interactive loop to answer questions from a single document |
| `3-semantic_search.py` | Semantic search to find the most relevant document for a query using sentence embeddings |
| `4-qa.py` | Complete system that uses semantic search to find relevant documents and BERT to extract answers |

## 🚀 Usage

### Basic Question Answering

To answer a specific question from a reference document:

```bash
./0-main.py
```

This will use the question-answering module to find an answer to a predefined question.

### Interactive Question Answering

For interactive question answering from a single reference document:

```bash
./2-main.py
```

Type your questions after the "Q:" prompt. The system will respond with answers if they can be found in the reference document, or with "Sorry, I do not understand your question" if no answer is found.

To exit, type "exit", "quit", "goodbye", or "bye" (case-insensitive).

### Multi-document Question Answering

For interactive question answering from multiple documents:

```bash
./4-main.py
```

This will search through all documents in the ZendeskArticles directory to find the most relevant document for each question, then extract answers from that document.

## 🧠 Model Information

The project uses the following models:

- **BERT** 📚: For extractive question answering
- **Universal Sentence Encoder** 🌐: For semantic search and document retrieval

## 💡 Examples

```
Q: When are PLDs?
A: on-site days from 9:00 am to 3:00 pm

Q: What are Mock Interviews?
A: help you train for technical interviews

Q: What does PLD stand for?
A: peer learning days

Q: goodbye
A: Goodbye
```