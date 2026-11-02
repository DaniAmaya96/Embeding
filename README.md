# Embeding

Semantic search over the Student Regulations of the Pontificia Universidad Javeriana.

This is a small project I did for my Advanced Linear Algebra course. The idea is simple: I take
the Student Regulations (Acuerdo 731 of 2022), cut it into articles, turn every article into a
vector, and then I can write a question in normal Spanish and get back the articles that are
closest to my question.

There is no chatbot here and no model writing answers. The notebook only does the search part
(what people call retrieval). It gives me back the articles ranked by similarity and I read them
myself. That was on purpose, I wanted to see the linear algebra, not to hide it behind an API.

## Files

| File | What it is |
|---|---|
| `Embeding.ipynb` | The whole pipeline, 15 cells |
| `CodeOfConductJaverianaUniversity.pdf` | The Student Regulations, 32 pages |

## Data

The PDF has 32 pages. Pages 1 and 2 are the Acuerdo that approves the regulation (the
"considerando" part), and the regulation itself starts on page 3. The regulation has 158
articles and they are written as `1.`, `2.`, ... `158.` at the beginning of a line. Article 158
is the last one, it is the one that says when the regulation enters into force.

## What the notebook does

1. Read the PDF with `pypdf` and keep one string per page.
2. Load the model `intfloat/multilingual-e5-small` with `sentence-transformers`. It is
   multilingual so it works in Spanish, and it accepts 512 tokens maximum.
3. Count the tokens of every page. Almost all the pages are between 540 and 714 tokens, so a
   page does not fit in the model. This is the reason why I need to cut the text.
4. Cut the text **by articles**, not by fixed windows of tokens. I use a regular expression to
   find the lines that start with the next expected number, so `1.` then `2.` then `3.` and so
   on. Lines in capital letters are saved as the section title, and every other line is added to
   the article that is currently open. This gives 158 articles.
5. Build the chunks. One chunk per article, except for the 5 articles that are longer than 450
   tokens (articles 8, 120, 121, 126 and 141). Those are cut in windows of 450 tokens with an
   overlap of 50 tokens, so a sentence that gets cut in the middle still appears complete in one
   of the two windows. This gives 167 chunks.
6. Add the prefix `passage: ` to every chunk and embed them. The E5 models are trained with
   `passage: ` for the documents and `query: ` for the question, so the prefixes are not
   decoration, the model expects them.
7. Ask the question, add the prefix `query: `, embed it.
8. Compare the question vector with the 167 chunk vectors using cosine similarity, and also
   compute the angle with `arccos` because that is what I wanted to show for the course.
9. Sort, take the top 5, and print a table with the article number, the page, the angle and the
   similarity.

## Output of the run that is saved in the notebook

```
Number of pages in the PDF: 32
Number of articles: 158
Number of chunks: 167
```

The question I used was "Cuantos crditos son matricula completa" (yes, with the typos, I left it
like that on purpose because it shows that the search still works). The first result is article
37, which is the only article in the whole regulation that talks about "matrícula completa". It
says that there are two modalities, full and half, and that the Consejo Directivo decides the
maximum number of credits of each one. So the regulation does not give a number, and the search
found the correct article anyway.

## How to run it

```bash
pip install sentence-transformers pypdf pandas
```

Then open `Embeding.ipynb` and run the cells in order. The first time it downloads the model
(around 470 MB). Cell 11 uses `input()`, so it will stop and wait for the question.

## Things I know are not perfect

- The regular expression for the section titles does not accept a dot, so the chapter titles
  that are written like `II. INGRESO A LA UNIVERSIDAD` are not detected as titles and they end
  up inside the text of the previous article. It happens 5 times (inside articles 8, 32, 59, 101
  and 111). It does not change the number of articles, only adds a few extra words.
- The token count of a long article is done before adding the header `Artículo N (SECTION): `,
  so the final chunks reach 470 tokens instead of 450. It is still under the 512 limit so
  nothing breaks.
- In cell 7 I create a list `chunks` that I never fill, that cell only prints the tokens per
  page. Cell 8 creates the real `chunks`.
- The model already normalizes the vectors, so the norms are 1 and the division by the norms in
  cell 12 does not change the result. I left it there because I wanted the formula explicit.
- The comparison is a `for` loop over the 167 chunks. With 167 it is instant, but for a big
  corpus the right way is one matrix product.
- The bars in the final table are scaled between the minimum and the maximum of the 5 rows, so
  the last row always looks empty even when its similarity is 0.85.

## Next steps

- Compare the article partition against windows of fixed size to see which one retrieves better.
- Try a bigger model (`multilingual-e5-base`) and see if the ranking changes.
- Replace the loop with a matrix product and measure the time.
