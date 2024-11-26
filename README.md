# NLP-DeepLearning-Models
This repository implements deep learning models for NLP tasks, including text classification with RNN, GRU, LSTM, and Bidirectional RNN, as well as text generation using a fine-tuned GPT2 transformer model, with a focus on Arabic text data.
# Deep Learning for NLP - Sequence Models and Transformers

## Repository Name: NLP-DeepLearning-Models

### Master: SIBD  
### Course: Deep Learning  
### Instructor: Pr. Elaachak Lotfi  
### University: Université Abdelmalek Essaadi  
### Faculty: Faculté des Sciences et Techniques de Tanger  
### Department: Département Génie Informatique  

---

## Table of Contents
1. [Introduction](#introduction)
2. [Part 1: Classification Task](#part-1-classification-task)
   - [Dataset Collection](#dataset-collection)
   - [Preprocessing Pipeline](#preprocessing-pipeline)
   - [Model Architectures](#model-architectures)
   - [Model Evaluation](#model-evaluation)
3. [Part 2: Transformer (Text Generation)](#part-2-transformer-text-generation)
   - [Fine-tuning GPT2](#fine-tuning-gpt2)
   - [Text Generation](#text-generation)
4. [Tools & Technologies](#tools-technologies)
5. [Learnings and Conclusion](#learnings-and-conclusion)
6. [References](#references)

---

## Introduction
This repository presents a deep learning project focused on Natural Language Processing (NLP) using PyTorch. The project covers two main tasks: 
1. **Text Classification** using sequence models like RNN, GRU, LSTM, and Bidirectional RNN.
2. **Text Generation** using the GPT2 transformer model for fine-tuning and text generation tasks.

---

## Part 1: Classification Task

### Dataset Collection
For this task, we used web scraping tools like Scrapy and BeautifulSoup to collect text data from various Arabic websites on a specific topic. The collected texts are associated with a relevance score between 0 and 10, which reflects the quality and relevance of the text.

Example Dataset:
| Text | Score |
|------|-------|
| Text 1 (Arabic Language) | 6 |
| Text 2 (Arabic Language) | 7.5 |

### Preprocessing Pipeline
We implemented an NLP preprocessing pipeline that includes:
- **Tokenization**: Breaking text into smaller pieces (tokens).
- **Stop Words Removal**: Eliminating common words that do not add value to the analysis.
- **Stemming & Lemmatization**: Reducing words to their base or root forms.
- **Discretization**: Converting continuous values (scores) into discrete categories if needed.

### Model Architectures
The models used for classification tasks include:
- **Recurrent Neural Networks (RNN)**
- **Bidirectional RNN**
- **Gated Recurrent Unit (GRU)**
- **Long Short-Term Memory (LSTM)**

We experimented with different hyperparameters and architectures to achieve the best model performance.

### Model Evaluation
Models were evaluated using standard metrics such as:
- **Accuracy**
- **Precision, Recall, F1-Score**
- **BLEU Score** for evaluating text generation quality (if applicable).

---

## Part 2: Transformer (Text Generation)

### Fine-tuning GPT2
We fine-tuned the GPT2 pre-trained transformer model on a customized dataset. This involved loading the pre-trained GPT2 model and modifying it according to our task requirements, such as adjusting the learning rate and the number of training epochs.

### Text Generation
After fine-tuning, the model was used to generate new paragraphs based on a given sentence or prompt. The generated text could mimic the style and topic of the dataset on which the model was trained.

---

## Tools & Technologies
- **Google Colab / Kaggle**: Used for running models and training tasks.
- **PyTorch**: Framework used to build deep learning models.
- **Transformers (Hugging Face)**: Used for working with pre-trained GPT2 model.
- **Scrapy / BeautifulSoup**: Libraries used for web scraping to collect text data.
- **GitHub**: Version control and project management.

---

## Learnings and Conclusion
During the course of this lab, I gained valuable experience in building and fine-tuning deep learning models for NLP tasks. Key takeaways include:
- Understanding the importance of data preprocessing and its impact on model performance.
- Gaining hands-on experience with sequence models such as RNN, LSTM, and GRU.
- Working with state-of-the-art transformer models like GPT2 for text generation.
- Learning how to fine-tune pre-trained models to adapt them to specific tasks.

Overall, this project enhanced my understanding of deep learning techniques in NLP and provided practical experience with cutting-edge models.

---

## References
- [Tutorial: Fine-tuning GPT2](https://gist.github.com/mf1024/3df214d2f17f3dcc56450ddf0d5a4cd7)
