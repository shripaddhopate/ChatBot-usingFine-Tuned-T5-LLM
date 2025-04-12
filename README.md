# ChatBot-usingFine-Tuned-T5-LLM
This project involves building an intelligent chatbot by fine-tuning the T5-small large language model architecture. The chatbot is designed to handle question answering and other NLP tasks by learning from a custom dataset generated from scanned book PDFs.
## 🧠 Overview:
This project involves building an intelligent chatbot by fine-tuning the T5-small large language model architecture. The chatbot is designed to handle question answering and other NLP tasks by learning from a custom dataset generated from scanned book PDFs.

## 🛠️ Key Components:
PDF to Image Conversion:
The project starts by converting educational or document-based PDFs into images to facilitate OCR-based text extraction.

- ### OCR using PaddleOCR:
High-accuracy text extraction is achieved using PaddleOCR, which efficiently processes scanned document images and extracts clean textual content.

- ### Data Transformation using Gemini LLM API:
The extracted raw text is then transformed into structured Question-Answer (Q&A) format using Google Gemini API. This helps in creating a high-quality dataset tailored for the QA task.

- ### Batch Processing & Post-Processing:
Extracted Q&A pairs are grouped into batches and undergo post-processing to clean, normalize, and enhance the textual data before training.

- ### Model Training - Fine-Tuning T5-small:
The T5-small architecture, a powerful encoder-decoder-based transformer model, is fine-tuned on the processed dataset. This model supports multitasking and is capable of handling:

- Question Answering

- Text Generation

- Summarization

- Sentiment Analysis

- Text Classification

### 🤖 Chatbot Integration:
The fine-tuned model is integrated into a web-based chatbot UI that allows users to interact and receive intelligent, context-aware responses based on the domain-specific knowledge learned during training.

![image](https://github.com/user-attachments/assets/239e08be-1248-4393-855b-e940eaeda453)


