---
title: Paper - Leveraging LLMs for SQL Generation in Computational Biology
draft: false
tags:
  - Projects
  - MachineLearning
---

# Abstract

Advancements in Large Language Models (LLMs) have revolutionized automatic code generation
and provide opportunities for database management. This project explores the use of LLM agents
for generating SQL queries, presenting an intuitive method for naive users to interact with complex
databases. We fine-tune models, such as Llama-2-7B and Mistral-7B, using Gretel AI’s text-to-SQL
dataset and employ chain-of-thought prompt engineering to produce a SQL agent that generates
accurate, concise SQL queries spanning multiple tables of a fake company’s MySQL database[3-
5]. We evaluate our agent using metrics including the query compilation accuracy, query output
accuracy, and query verbosity. Results show that the combination of prompt engineering and fine-
tuning produces SQL agents with superior evaluation metrics than agents produced using either
method alone. Our lightweight, fine-tuned SQL agent can translate complex user questions into
concise, accurate queries and serves as a useful database tool for non-technical users. In terms of
applications in computational biology, medical records and databases, such as the AoU database,
hold critical information that can be hard to decipher. Creating domain-specific agents for SQL
generation can help in the space of biology due to its ability to ease data collection and aggregation.

## Full paper embedded below

![[Leveraging LLMs for SQL Generation in Computational Biology.pdf]]
