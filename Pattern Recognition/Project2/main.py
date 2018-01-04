import glob
from keyword_spotter import KeywordSpotter
from evaluation import Evaluation

image_paths = []
train = ["270", "271", "272", "273", "274", "275", "276", "277", "278", "279"]
# train = ["274"]
for i in train:
    image_paths += glob.glob(f"ground-truth/word-images_preprocessed/{i}-*.png")

query_paths = []
valid = ["300", "301", "302", "303", "304"]
for i in valid:
    query_paths += glob.glob(f"ground-truth/word-images_preprocessed/{i}-*.png")

# query_paths = ["ground-truth/word-images_preprocessed/303-16-06.png"]

s = KeywordSpotter(image_paths)

for query_image in query_paths:
    print(f"Spotting {query_image}")

    retrieved = s.spot(query_image)

    e = Evaluation(
        all_elements=s.images,
        query_element=s.query_image,
        retrieved_elements=retrieved
    )

    print("Precision: " + str(e.precision()))
    print("Recall: " + str(e.recall()))
    print("\n")