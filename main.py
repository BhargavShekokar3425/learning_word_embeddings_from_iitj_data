"""
Main Script for Word2Vec Assignment
This is the entry point for running the entire assignment.
Run this script to execute all tasks from data collection to visualization.

Author: Bhargav Shekokar (B23CS1008)
Course: NLU Assignment 2

Usage:
    python main.py              # Run everything
    python main.py --task 1     # Run only Task 1 (data collection & preprocessing)
    python main.py --task 2     # Run only Task 2 (model training)
    python main.py --task 3     # Run only Task 3 (semantic analysis)
    python main.py --task 4     # Run only Task 4 (visualization)
"""

import argparse
import os
import sys
import time
from datetime import datetime

# import our modules
from data_collector import collect_data
from preprocessor import preprocess_data, TextPreprocessor
from word2vec_trainer import Word2VecTrainer, train_models
from semantic_analysis import SemanticAnalyzer, run_semantic_analysis
from visualizer import EmbeddingVisualizer, create_visualizations


def print_banner():
    """
    Prints a nice banner because why not.
    Makes the output look more organized.
    """
    banner = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║       WORD2VEC ASSIGNMENT - IIT JODHPUR CORPUS                   ║
    ║       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                   ║
    ║       Author: Bhargav Shekokar (B23CS1008)                       ║
    ║       Course: Natural Language Understanding                      ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def task1_data_preparation():
    """
    Task 1: Dataset Preparation
    - Collect data from IIT Jodhpur sources
    - Preprocess and clean the text
    - Generate statistics and word cloud
    """
    print("\n" + "=" * 70)
    print("TASK 1: DATASET PREPARATION")
    print("=" * 70)
    
    # Step 1: Collect data
    print("\n[Step 1/3] Collecting data from IIT Jodhpur sources...")
    corpus_path = collect_data(use_sample=True)  # using sample for reliability
    
    # Step 2: Preprocess data
    print("\n[Step 2/3] Preprocessing collected data...")
    sentences, stats = preprocess_data()
    
    # Step 3: Generate visualizations
    print("\n[Step 3/3] Generating word cloud and frequency plots...")
    # (already done in preprocess_data)
    
    print("\n✓ Task 1 Complete!")
    print(f"  - Documents processed: {stats['total_documents']}")
    print(f"  - Total tokens: {stats['total_tokens']}")
    print(f"  - Vocabulary size: {stats['vocabulary_size']}")
    print(f"  - Word cloud saved to: ./outputs/wordcloud.png")
    print(f"  - Cleaned corpus saved to: ./data/processed/cleaned_corpus.txt")
    print(f"  - Assignment corpus saved to: ./data/processed/corpus.txt")
    
    return sentences, stats


def task2_model_training():
    """
    Task 2: Model Training
    - Train CBOW and Skip-gram models
    - Experiment with different hyperparameters
    - Report results
    """
    print("\n" + "=" * 70)
    print("TASK 2: MODEL TRAINING")
    print("=" * 70)
    
    # check if preprocessed data exists
    corpus_path = "./data/processed/cleaned_corpus.txt"
    if not os.path.exists(corpus_path):
        print("\nPreprocessed corpus not found!")
        print("Running Task 1 first...")
        task1_data_preparation()
    
    print("\nTraining Word2Vec models with various configurations...")
    print("This might take a few minutes...\n")
    
    # train all models and run experiments
    cbow_model, skipgram_model = train_models(corpus_path)
    
    print("\n✓ Task 2 Complete!")
    print(f"  - Models saved to: ./models/")
    print(f"  - Experiment results saved to: ./outputs/experiment_results.json")
    
    return cbow_model, skipgram_model


def task3_semantic_analysis(cbow_model=None, skipgram_model=None):
    """
    Task 3: Semantic Analysis
    - Find similar words using cosine similarity
    - Perform analogy experiments
    """
    print("\n" + "=" * 70)
    print("TASK 3: SEMANTIC ANALYSIS")
    print("=" * 70)
    
    # check if models exist
    if cbow_model is None and skipgram_model is None:
        cbow_path = "./models/cbow_dim100_win5_neg5.model"
        sg_path = "./models/skipgram_dim100_win5_neg5.model"
        
        if not os.path.exists(cbow_path) and not os.path.exists(sg_path):
            print("\nTrained models not found!")
            print("Running Task 2 first...")
            cbow_model, skipgram_model = task2_model_training()
    
    print("\nAnalyzing semantic relationships...")
    
    # run semantic analysis
    analyzer = run_semantic_analysis(cbow_model, skipgram_model)
    
    print("\n✓ Task 3 Complete!")
    print(f"  - Similarity results saved to: ./outputs/similarity_results.json")
    print(f"  - Analogy results saved to: ./outputs/analogy_results.json")
    
    return analyzer


def task4_visualization(cbow_model=None, skipgram_model=None):
    """
    Task 4: Visualization
    - Project embeddings to 2D using PCA and t-SNE
    - Visualize semantic clusters
    - Compare CBOW and Skip-gram
    """
    print("\n" + "=" * 70)
    print("TASK 4: VISUALIZATION")
    print("=" * 70)
    
    # check if models exist
    if cbow_model is None and skipgram_model is None:
        cbow_path = "./models/cbow_dim100_win5_neg5.model"
        sg_path = "./models/skipgram_dim100_win5_neg5.model"
        
        if not os.path.exists(cbow_path) and not os.path.exists(sg_path):
            print("\nTrained models not found!")
            print("Running Task 2 first...")
            cbow_model, skipgram_model = task2_model_training()
    
    print("\nCreating visualizations...")
    
    # create visualizations
    visualizer = create_visualizations(cbow_model, skipgram_model)
    
    print("\n✓ Task 4 Complete!")
    print(f"  - Visualizations saved to: ./outputs/visualizations/")
    
    return visualizer


def run_all_tasks():
    """
    Runs all tasks in sequence.
    This is the default when running main.py without arguments.
    """
    start_time = time.time()
    
    # Task 1: Data Preparation
    sentences, stats = task1_data_preparation()
    
    # Task 2: Model Training
    cbow_model, skipgram_model = task2_model_training()
    
    # Task 3: Semantic Analysis
    analyzer = task3_semantic_analysis(cbow_model, skipgram_model)
    
    # Task 4: Visualization
    visualizer = task4_visualization(cbow_model, skipgram_model)
    
    total_time = time.time() - start_time
    
    # print final summary
    print("\n" + "=" * 70)
    print("ALL TASKS COMPLETED!")
    print("=" * 70)
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    
    print("\nOutput Files Generated:")
    print("-" * 40)
    print("📁 data/")
    print("   └── raw/              - Raw collected documents")
    print("   └── processed/        - Cleaned corpus and vocabulary")
    print("       └── corpus.txt    - Assignment corpus file")
    print("📁 models/               - Trained Word2Vec models")
    print("📁 outputs/")
    print("   └── wordcloud.png     - Word cloud visualization")
    print("   └── freq_dist.png     - Word frequency distribution")
    print("   └── experiment_results.json")
    print("   └── similarity_results.json")
    print("   └── analogy_results.json")
    print("   └── visualizations/   - PCA and t-SNE plots")
    
    print("\n" + "=" * 70)
    print("Assignment execution complete!")
    print("=" * 70)


def main():
    """
    Main entry point.
    Parses arguments and runs requested tasks.
    """
    parser = argparse.ArgumentParser(
        description='Word2Vec Assignment - IIT Jodhpur Corpus',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py              # Run all tasks
    python main.py --task 1     # Run only data preparation
    python main.py --task 2     # Run only model training
    python main.py --task 3     # Run only semantic analysis
    python main.py --task 4     # Run only visualization
    python main.py --task 1 2   # Run tasks 1 and 2
        """
    )
    
    parser.add_argument(
        '--task', '-t',
        type=int,
        nargs='+',
        choices=[1, 2, 3, 4],
        help='Specific task(s) to run (1-4). If not specified, runs all tasks.'
    )
    
    parser.add_argument(
        '--use-live-data',
        action='store_true',
        help='Use live web scraping instead of sample data (may be slower/unreliable)'
    )
    
    args = parser.parse_args()
    
    # print banner
    print_banner()
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # create necessary directories
    os.makedirs("./data/raw", exist_ok=True)
    os.makedirs("./data/processed", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./outputs", exist_ok=True)
    
    # run requested tasks
    if args.task is None:
        # run all tasks
        run_all_tasks()
    else:
        # run specific tasks
        cbow_model, skipgram_model = None, None
        
        for task_num in sorted(args.task):
            if task_num == 1:
                task1_data_preparation()
            elif task_num == 2:
                cbow_model, skipgram_model = task2_model_training()
            elif task_num == 3:
                task3_semantic_analysis(cbow_model, skipgram_model)
            elif task_num == 4:
                task4_visualization(cbow_model, skipgram_model)
    
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
