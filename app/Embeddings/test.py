from app.Embeddings.embeddings import get_embedding_model


model = get_embedding_model()

text = "I need lightweight running shoes"

vector = model.embed_query(text)

print("Vector dimensions:", len(vector))
print("First 5 values:", vector[:5])

assert len(vector) == 384