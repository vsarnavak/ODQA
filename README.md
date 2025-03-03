## Overview
This is an open-domain question answering system for the Tamil language. The system takes a user-input question in Tamil and retrieves relevant information from Wikipedia to generate a context-aware answer.

## Features
- Accepts Tamil language questions.
- Processes and tokenizes the question to improve search accuracy.
- Extracts relevant information from Wikipedia articles.
- Uses a BERT-based model to refine and generate the final answer.
- Provides a Flask web interface for user-friendly interaction.

## How It Works
1. The user enters a Tamil question in the web interface.
2. The system tokenizes and normalizes the question.
3. It searches for relevant Wikipedia articles.
4. The retrieved content is processed and passed to the BERT-based model.
5. The model generates a refined answer.
6. The system displays both the context and the final extracted answer.

## Example Questions
- **"செயற்கை நுண்ணறிவு என்றால் என்ன?"** (What is artificial intelligence?)
- **"திருக்குறள் என்றால் என்ன?"** (What is Thirukkural?)
- **"பெரியார் எப்போது பிறந்தார்?"** (When was Periyar born?)


