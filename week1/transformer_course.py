from transformers import pipeline

classifier = pipeline('sentiment-analysis')
classifier("I have finally started learning LLMs")

classifier = pipeline("zero-shot-classification")
classifier(
    "Iran has taken control of the hormoz Strait",
    candidate_labels=["education", "politics", "business"],
)

generator = pipeline("text-generation")
generator("In this course, we will teach you how to")