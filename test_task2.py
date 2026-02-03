from task1 import load_dataset, list_datasets
from task2 import ask_llm

# Charge dataset test
df = load_dataset("iris")
columns = df.columns.tolist()
print("Colonnes :", columns)

# Exemple de FDs fictives
fds = [
    (["sepal_length"], "sepal_width"),
    (["petal_length", "petal_width"], "species"),
    (["sepal_length", "sepal_width"], "petal_length")
]

for lhs, rhs in fds:
    fd_str = f"{', '.join(lhs)} → {rhs}"
    label = ask_llm(fd_str, "iris")
    print(fd_str, "→ LLM label:", label)
