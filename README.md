## Javeriana Semantic Search

A Python notebook for searching the Student Regulations of the Pontificia Universidad Javeriana (Acuerdo 731 of 2022). I developed it for my Advanced Linear Algebra course to apply text embeddings and vector similarity to an information retrieval problem.

You can enter a question in Spanish and retrieve the most relevant passages from the document.

# How it works

The notebook extracts the PDF text and separates its 158 articles. Longer articles are divided into overlapping chunks, producing 167 passages.

It uses intfloat/multilingual-e5-small from Hugging Face to generate embeddings for the passages and the question. It then calculates cosine similarity and the angle between vectors, returning the five closest passages with their article number, page, and scores.

For example, the question “¿Cuántos créditos son matrícula completa?” retrieves Article 37, which discusses full and half enrollment. The results are excerpts from the regulation, which the user reads directly.

# Files

Embeding.ipynb: text extraction, article segmentation, embeddings, and search.

CodeOfConductJaverianaUniversity.pdf: source document.

# How to run

Install the dependencies:

pip install sentence-transformers pypdf pandas

Open Embeding.ipynb and run the cells in order. The model downloads on the first run. When prompted, enter your question in Spanish.
