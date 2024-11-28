from src.naive_bayes_binarization import main as binarization_main
from src.naive_bayes_ngram import main as ngram_main
from src.naive_bayes_tfidf import main as tfidf_main


def run_all_classifiers():
    print("\nRunning Naive Bayes with Binarization...")
    binarization_main()

    print("\nRunning Naive Bayes with N-Grams...")
    ngram_main()

    print("\nRunning Naive Bayes with TF-IDF...")
    tfidf_main()


def main():
    print("\nChoose an option:")
    print("1: Run Naive Bayes with Binarization")
    print("2: Run Naive Bayes with N-Grams")
    print("3: Run Naive Bayes with TF-IDF")
    print("4: Run all classifiers")

    choice = input("Enter your choice (1/2/3/4): ").strip()

    if choice == "1":
        binarization_main()
    elif choice == "2":
        ngram_main()
    elif choice == "3":
        tfidf_main()
    elif choice == "4":
        run_all_classifiers()
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
