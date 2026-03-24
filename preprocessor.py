"""
Text Preprocessing Module for Word2Vec Assignment
=================================================
This handles all the messy text cleaning stuff.
Tokenization, lowercasing, removing junk, etc.

Author: Bhargav Shekokar (B23CS1008)
Course: NLU Assignment 2
"""

import re
import os
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np

# download nltk data if not already there
# this is annoying but necessary
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)


class TextPreprocessor:
    """
    Does all the text cleaning and preprocessing.
    Takes raw messy text and makes it nice and clean for Word2Vec.
    """
    
    def __init__(self, remove_stopwords=False):
        """
        Initialize the preprocessor.
        
        Args:
            remove_stopwords: Whether to remove common words like 'the', 'is', etc.
                              For Word2Vec, we usually keep them because they
                              help with context. Set to False by default.
        """
        self.remove_stopwords = remove_stopwords
        
        # get english stopwords if we need them
        try:
            self.stopwords = set(stopwords.words('english'))
        except:
            nltk.download('stopwords', quiet=True)
            self.stopwords = set(stopwords.words('english'))
        
        # keep track of stats
        self.stats = {
            'total_documents': 0,
            'total_tokens': 0,
            'vocabulary_size': 0,
            'word_frequencies': Counter()
        }
        
    def clean_text(self, text):
        """
        Basic text cleaning - removes all the garbage.
        This is where we get rid of HTML artifacts, weird characters, etc.
        """
        # first, let's handle some common Unicode issues
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        # remove URLs - they're not useful for word embeddings
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # remove phone numbers (various formats)
        text = re.sub(r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}', '', text)
        
        # remove special characters but keep letters, numbers, and basic punctuation
        # we need spaces and sentence boundaries
        text = re.sub(r'[^\w\s\.\,\!\?\-]', ' ', text)
        
        # convert multiple spaces to single space
        text = re.sub(r'\s+', ' ', text)
        
        # remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def remove_non_english(self, text):
        """
        Removes non-English characters.
        The assignment says to only keep English text.
        """
        # keep only ASCII characters basically
        # this removes Hindi, special characters, etc.
        cleaned = re.sub(r'[^\x00-\x7F]+', ' ', text)
        return cleaned
    
    def tokenize(self, text):
        """
        Tokenizes text into words.
        Also does lowercasing because we want 'Research' and 'research' to be the same.
        """
        # lowercase everything first
        text = text.lower()
        
        # try nltk tokenizer, fall back to simple split if it fails
        try:
            tokens = word_tokenize(text)
        except:
            # simple fallback - just split on whitespace and punctuation
            tokens = re.findall(r'\b\w+\b', text)
        
        # filter out tokens that are just numbers or single characters
        # (except meaningful single chars like 'a', 'i')
        filtered_tokens = []
        for token in tokens:
            # skip if it's just punctuation
            if re.match(r'^[^\w]+$', token):
                continue
            # skip if it's just numbers
            if re.match(r'^\d+$', token):
                continue
            # skip single characters (except 'a' and 'i' maybe)
            if len(token) == 1 and token not in ['a', 'i']:
                continue
            
            filtered_tokens.append(token)
        
        # optionally remove stopwords
        if self.remove_stopwords:
            filtered_tokens = [t for t in filtered_tokens if t not in self.stopwords]
        
        return filtered_tokens
    
    def process_document(self, text):
        """
        Full preprocessing pipeline for a single document.
        Returns list of tokens ready for Word2Vec.
        """
        # step 1: remove non-english stuff
        text = self.remove_non_english(text)
        
        # step 2: basic cleaning
        text = self.clean_text(text)
        
        # step 3: tokenize
        tokens = self.tokenize(text)
        
        return tokens
    
    def process_corpus(self, input_path, output_path="./data/processed"):
        """
        Processes entire corpus from input file or directory.
        Creates cleaned corpus file and computes statistics.
        
        Args:
            input_path: Path to raw corpus file or directory
            output_path: Where to save processed data
        """
        os.makedirs(output_path, exist_ok=True)
        
        all_tokens = []
        all_sentences = []  # for Word2Vec we need sentences (lists of words)
        
        # check if input is file or directory
        if os.path.isfile(input_path):
            files = [input_path]
        else:
            files = [os.path.join(input_path, f) for f in os.listdir(input_path) 
                     if f.endswith('.txt')]
        
        self.stats['total_documents'] = len(files)
        print(f"Processing {len(files)} documents...")
        
        for filepath in files:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = f.read()
            
            # split into sentences first (for Word2Vec training)
            try:
                sentences = sent_tokenize(raw_text)
            except:
                # fallback: split on periods
                sentences = raw_text.split('.')
            
            for sent in sentences:
                tokens = self.process_document(sent)
                if len(tokens) >= 3:  # skip very short sentences
                    all_sentences.append(tokens)
                    all_tokens.extend(tokens)
        
        # calculate statistics
        self.stats['total_tokens'] = len(all_tokens)
        self.stats['word_frequencies'] = Counter(all_tokens)
        self.stats['vocabulary_size'] = len(self.stats['word_frequencies'])
        
        # save processed corpus
        # format: one sentence per line, words separated by spaces
        corpus_path = os.path.join(output_path, "cleaned_corpus.txt")
        with open(corpus_path, 'w', encoding='utf-8') as f:
            for sentence in all_sentences:
                f.write(' '.join(sentence) + '\n')
        
        print(f"Processed corpus saved to {corpus_path}")
        print(f"Total sentences: {len(all_sentences)}")
        
        # save vocabulary
        vocab_path = os.path.join(output_path, "vocabulary.txt")
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for word, count in self.stats['word_frequencies'].most_common():
                f.write(f"{word}\t{count}\n")
        
        print(f"Vocabulary saved to {vocab_path}")
        
        return all_sentences, corpus_path
    
    def print_stats(self):
        """
        Prints dataset statistics.
        Useful for the report.
        """
        print("\n" + "=" * 50)
        print("DATASET STATISTICS")
        print("=" * 50)
        print(f"Total Documents: {self.stats['total_documents']}")
        print(f"Total Tokens: {self.stats['total_tokens']}")
        print(f"Vocabulary Size: {self.stats['vocabulary_size']}")
        print(f"\nTop 20 Most Frequent Words:")
        print("-" * 30)
        for word, count in self.stats['word_frequencies'].most_common(20):
            print(f"  {word}: {count}")
        print("=" * 50 + "\n")
        
        return self.stats
    
    def generate_wordcloud(self, save_path="./outputs/wordcloud.png"):
        """
        Generates a word cloud visualization.
        Everyone loves word clouds - they look cool in reports.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if not self.stats['word_frequencies']:
            print("No word frequencies available. Process corpus first!")
            return
        
        # create word cloud
        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            colormap='viridis',  # nice color scheme
            max_words=150,
            relative_scaling=0.5,
            min_font_size=10
        ).generate_from_frequencies(self.stats['word_frequencies'])
        
        # plot and save
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud - IIT Jodhpur Corpus', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Word cloud saved to {save_path}")
    
    def plot_frequency_distribution(self, top_n=30, save_path="./outputs/freq_dist.png"):
        """
        Plots word frequency distribution.
        Bar chart showing most common words.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if not self.stats['word_frequencies']:
            print("No word frequencies available. Process corpus first!")
            return
        
        # get top N words
        top_words = self.stats['word_frequencies'].most_common(top_n)
        words, counts = zip(*top_words)
        
        # create bar plot
        plt.figure(figsize=(14, 8))
        bars = plt.bar(range(len(words)), counts, color='steelblue', edgecolor='navy')
        
        # add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontsize=8)
        
        plt.xticks(range(len(words)), words, rotation=45, ha='right')
        plt.xlabel('Words', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Top {top_n} Most Frequent Words in IIT Jodhpur Corpus', 
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Frequency distribution plot saved to {save_path}")


def preprocess_data(input_path="./data/raw", output_path="./data/processed"):
    """
    Main function to preprocess the collected data.
    Call this after running the data collector.
    """
    preprocessor = TextPreprocessor(remove_stopwords=False)
    
    # process the corpus
    sentences, corpus_path = preprocessor.process_corpus(input_path, output_path)
    
    # print statistics
    stats = preprocessor.print_stats()
    
    # generate visualizations
    preprocessor.generate_wordcloud()
    preprocessor.plot_frequency_distribution()
    
    return sentences, stats


if __name__ == "__main__":
    # run preprocessing on collected data
    preprocess_data()
