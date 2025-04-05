from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



load_dotenv()

embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

docs = [
    "Virat Kohli is an Indian cricketer and former captain of the Indian national team. He is considered one of the best batsmen in the world.",
    "Steve Smith is an Australian cricketer known for his unorthodox batting style and exceptional test match batting average.",
    "Kane Williamson is a New Zealand cricketer who is the current captain of the New Zealand national team in all formats.",
    "Ben Stokes is an English cricketer who is known for his aggressive batting style and ability to perform in high-pressure situations.",
    "Babar Azam is a Pakistani cricketer who is considered one of the best batsmen in modern cricket across all formats."
]
queary="Who is ben stokes"

vector=embeddings.embed_documents(docs)
question=embeddings.embed_query(queary)

similarity_search=cosine_similarity([question],vector)[0]#esmain hamaesaha 2d vector atay hain
#first approach gives the max
# max_sim=0
# most_relevant_doc=None
# for i,sim in enumerate(similarity_search):
#     if max_sim < sim:
#         max_sim=sim
#         most_relevant_doc=docs[i]

#second approach used by sorting


i,sim=sorted(list(enumerate(similarity_search)),key=lambda x:x[1])[-1]

#explaination
# Let's say similarity_search = [0.2, 0.5, 0.3, 0.1, 0.8]
# And docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]

# Step 1: enumerate(similarity_search)
# Creates: [(0, 0.2), (1, 0.5), (2, 0.3), (3, 0.1), (4, 0.8)]

# Step 2: list(enumerate(similarity_search))
# Still: [(0, 0.2), (1, 0.5), (2, 0.3), (3, 0.1), (4, 0.8)]

# Step 3: sorted(..., key=lambda x:x[1])
# Sorts by similarity score:
# [(3, 0.1), (0, 0.2), (2, 0.3), (1, 0.5), (4, 0.8)]

# Step 4: [-1]
# Gets: (4, 0.8)

# Step 5: i, sim = ...
# i = 4
# sim = 0.8


print(f"\nQueary:{queary}")
print(f"doc:{docs[i]}")
print(sim)

# print(f"\nQuery: {queary}")
# print(f"Most relevant document: {most_relevant_doc}")
# print(f"Similarity score: {max_sim:.4f}")





