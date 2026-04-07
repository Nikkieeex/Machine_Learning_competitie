from model import model_factory

# Maak model (dit traint automatisch)
model = model_factory()

# Voorspel op een testbestand
preds = model.predict("test.csv")

print(preds)
