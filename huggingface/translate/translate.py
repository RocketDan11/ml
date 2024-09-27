# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("translation", model="google-t5/t5-base")