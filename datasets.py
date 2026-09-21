"""
CLI Entry point for the multi-source NLP Corpus Builder.
Delegates to src.data.fetcher for Gutenberg, Gutendex, Wikipedia, Fandom, and HF datasets.
"""
import sys
from src.data.fetcher import main

if __name__ == "__main__":
    main()
