from task1 import load_dataset, list_datasets

from task1 import list_datasets, load_dataset

print(list_datasets())

df = load_dataset("iris")
print(df.head())


def test_list_datasets():
    datasets = list_datasets()
    assert isinstance(datasets, list)
    assert len(datasets) > 0
    print("✅ list_datasets OK")

def test_load_dataset():
    datasets = list_datasets()
    ds = load_dataset(datasets[0])
    assert ds is not None
    print("✅ load_dataset OK")

if __name__ == "__main__":
    test_list_datasets()
    test_load_dataset()
    print("🎉 Tous les tests Task 1 sont OK")
