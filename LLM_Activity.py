# %% [markdown]
# ## Ollama LLM Analysis
# LLM Analytics Pipeline Assignment (Ollama Version)
# This verison is for local setup using Ollama + LangChain in VS Code

# %% [markdown]
# ### Step 1: Import Required Packages
# You may have to install new packages to your conda environment

# %%
from langchain_ollama.llms import OllamaLLM
import nltk
import wikipediaapi
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import FreqDist, pos_tag, word_tokenize
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt
import pandas as pd

# Download NLTK tools
nltk.download('punkt')
nltk.download('vader_lexicon')

# %% [markdown]
# ### Force install  langchain-ollama in the exact Python environment's notebook.

# %%
import sys
!{sys.executable} -m pip install langchain-ollama



# %% [markdown]
# ### Step 2: Connect to Local Mistral Model via Ollama 

# %%
from langchain_ollama.llms import OllamaLLM

llm = OllamaLLM(model="mistral")
response = llm.invoke("What years was Boston College Men's Hockey the runner-up for the National Championship?")
print(response)

# Save output
gen__string = response

# %% [markdown]
# ### Step 3: Analyze Sentiment using NLTK

# %%
import nltk
nltk.download('vader_lexicon')


# %%
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize the analyzer
analyzer = SentimentIntensityAnalyzer()

# Use the LLM response as input (you already generated this in Step 2)
sentiment = analyzer.polarity_scores(response)
print("Sentiment Scores:", sentiment)


# %% [markdown]
# Sentiment Analysis Interpretation:
# The sentiment scores of the model shows a moderately negative tone, with a compound score of -0.5574. Although, the response was mostly neutral (71%), the negative elements (17%) likely stem from emotionally charged words like "loss," "defeated," and "heartbreaking." This aligns given the topic involves Boston College losing championship games. 

# %% [markdown]
# ### Step 4: Calculate the BLEU Score
# "How close was the model's response to the actual facts?"

# %% [markdown]
# Step 4.1 Install & Import Wikipedia API

# %%
import sys
!{sys.executable} -m pip install wikipedia-api


# %%
import wikipediaapi


# %%
import wikipediaapi

wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='LLM-Analytics-Student/1.0 (leegcp@bc.edu; Boston College)',
)



# %%
page = wiki_wiki.page("Boston College Eagles men's ice hockey")
section = page.section_by_title("National Championships")

if section:
    print("Wikipedia Section Text:\n", section.text)
    reference = section.text.split()
else:
    print("Section not found.")
    reference = []


# %% [markdown]
# ### Use the LLM to generate output: ###

# %%
generated = response.split()


# %% [markdown]
# ### Use the truth from Wikipedia ###

# %%
reference = section.text.split()

# %% [markdown]
# ### Run the BLEU score comparison ###

# %%
from nltk.translate.bleu_score import sentence_bleu

score = sentence_bleu([reference], generated)
print("BLEU Score:", score)


# %% [markdown]
# ### Step 5 – BLEU Score Interpretation
# 
# The BLEU score between the model's generated output and the Wikipedia reference text was approximately **6.44 × 10⁻²³²**, which is effectively zero. This extremely low score indicates that the model's response shared very little structural similarity with the actual runner-up years listed on Wikipedia.
# 
# However, this doesn’t necessarily mean the model’s output was completely wrong — just that it presented the information in a different format (narrative style vs. simple list). The model provided years like 2010, 2012, and 2018, which are factually incorrect according to the Wikipedia entry.
# 
# This result shows that while the LLM was coherent and stylistic in tone, it was not reliable for factual recall in this specific case. The BLEU score accurately reflects that discrepancy in content accuracy.

# %% [markdown]
# ### Step 6: Improve the Prompt & Try Again
#  "What years was Boston College Men's Hockey the runner-up for the National Championship?"
# 
# New Prompt:
# "List only the years that Boston College Men's Hockey finished as the runner-up in the NCAA Division I Men's Ice Hockey Championship. Provide only the years, separated by commas. Do not include extra text or context."

# %%
improved_prompt = (
    "List only the years that Boston College Men's Hockey finished as the runner-up "
    "in the NCAA Division I Men's Ice Hockey Championship. "
    "Provide only the years, separated by commas. Do not include extra text or context."
)

improved_response = llm.invoke(improved_prompt)
print("Improved Response:\n", improved_response)

# Prepare for BLEU scoring
generated2 = improved_response.split()


# %%
score2 = sentence_bleu([reference], generated2)
print("Improved BLEU Score:", score2)


# %% [markdown]
# ### Step 7: BLEU Score Comparison & Prompt Engineering Impact
# 
# After improving the prompt to specifically request only the runner-up years separated by commas, the model returned:  
# **1949, 2012, 2018**
# 
# This was more structured and aligned in format with the reference data, the  content remained incorrect as these years are not consistent with the verified data from Wikipedia.
# 
# The BLEU score of the improved output was **0**, which matches the initial attempt and reflects the fact that the model did not produce any matching n-grams (even after formatting improvements).
# 
# This shows that prompt engineering helped with format clarity, but it did not help the model retrieve more accurate content. The takeaway is that prompt quality alone isn’t enough when the model lacks factual recall. This is common in many LLMs that aren’t retrieval-augmented or fine-tuned on niche topics.

# %% [markdown]
# ### Step 8: What is RAG (Retrieval-Augmented Generation?)
# 
# RAG stands for **Retrieval-Augmented Generation**, a technique that combines a language model (LLM) with an external data source like Wikipedia or a custom knowledge base. Instead of relying on the model’s internal memory, RAG retrieves relevant documents in real-time and feeds them into the prompt, improving accuracy and grounding the response in factual content.

# %%
# Sample RAG pipeline with LangChain (DO NOT RUN)

from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI

# Load your FAISS knowledge base
db = FAISS.load_local("my_faiss_index", OpenAIEmbeddings())
retriever = db.as_retriever()

# Create the RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=retriever
)

# Ask a question grounded in documents
result = rag_chain.run("What years was Boston College runner-up?")
print(result)


# %% [markdown]
# ### Step 9: Foundation Model vs. Fine-Tuning
# 
# A **foundation model** is a large, pre-trained language model trained on broad data (like GPT-3, Mistral, or LLaMA). **Fine-tuning** is the process of training that model further on a smaller, specialized dataset to improve its performance on specific tasks. While foundation models are general-purpose, fine-tuning customizes them for niche use cases like legal writing, medical Q&A, or historical sports stats.

# %% [markdown]
# ### Step 10: Hugging Face Fine-Tuning Sample Code (Do Not Run)

# %%
# Sample fine-tuning pipeline using Hugging Face (DO NOT RUN)

from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# Load dataset and tokenizer
dataset = load_dataset("imdb")  # Example: IMDB sentiment classification
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Preprocess the data
def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding=True)

tokenized_dataset = dataset.map(preprocess, batched=True)

# Load model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Set training parameters
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Create Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].shuffle().select(range(1000)),
    eval_dataset=tokenized_dataset["test"].shuffle().select(range(1000)),
)

# Train the model
# trainer.train()


# %% [markdown]
# ## 📘 Graded Assignment: 8 Questions
# 
# ### 1. Interpret the BLEU score. Is this a good score? Should we trust it?
# 
# The BLEU score was approximately **6.44 × 10⁻²³²**, which is effectively zero. This indicates the model’s output had almost no overlap with the reference text from Wikipedia. It is **not a good score**, and we should **not trust** the factual accuracy of the model's response in this case.
# 
# ---
# 
# ### 2. Update the LangChain prompt and improve it. Print and save as `generated_2`.
# 
# Updated the prompt to:  
# > *"List only the years that Boston College Men's Hockey finished as the runner-up in the NCAA Division I Men's Ice Hockey Championship. Provide only the years, separated by commas. Do not include extra text or context."*
# 
# Saved the improved output to `generated_2`.
# 
# ---
# 
# ### 3. Discuss any aspects of prompt engineering you used to improve the output.
# 
# To improve the output, I clarified **exactly what format I wanted** (a list of years, no extra text). I also specified “runner-up” and added constraints like “commas only” and “no context” to avoid narrative answers. This helped the model produce a cleaner result, even if the facts were still inaccurate.
# 
# ---
# 
# ### 4. Calculate the new BLEU score. Did it go up or down?
# 
# The new BLEU score was also **0**, meaning that even though the formatting improved, the content remained inaccurate. The score did **not improve**, which shows that formatting can help guide output — but **can’t fix factual recall** in models that lack access to reliable data.
# 
# ---
# 
# ### 5. What is Retrieval Augmented Generation in your own words?
# 
# RAG stands for **Retrieval-Augmented Generation**. It combines an LLM with a search or retrieval component (like Wikipedia), so the model can reference **real facts in real-time** instead of relying only on its internal memory. This helps improve factual accuracy.
# 
# ---
# 
# ### 6. Paste example RAG code (do not run).
# 
# ```python
# # Sample RAG pipeline with LangChain (do not run)
# from langchain.chains import RetrievalQA
# from langchain.vectorstores import FAISS
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.llms import OpenAI
# 
# db = FAISS.load_local("my_faiss_index", OpenAIEmbeddings())
# retriever = db.as_retriever()
# 
# rag_chain = RetrievalQA.from_chain_type(
#     llm=OpenAI(),
#     chain_type="stuff",
#     retriever=retriever
# )
# 
# result = rag_chain.run("What years was Boston College runner-up?")
# print(result)
# 

# %% [markdown]
# 7. In 3 sentences or fewer, explain foundation model vs. fine-tuning.
# 
# A foundation model is a large, general-purpose model trained on diverse data (like GPT-4 or Mistral). Fine-tuning means training that model further on a specific dataset to specialize it. Foundation models are broad; fine-tuned models are focused and task-specific.

# %% [markdown]
# 8. Paste example Hugging Face fine-tuning code (do not run).

# %%
# Hugging Face fine-tuning sample (do not run)
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding=True)

tokenized_dataset = dataset.map(preprocess, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].shuffle().select(range(1000)),
    eval_dataset=tokenized_dataset["test"].shuffle().select(range(1000)),
)

# trainer.train()



