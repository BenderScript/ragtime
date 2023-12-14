class SimpleRetriever:
    def __init__(self, documents):
        """
        Initializes the SimpleRetriever with a list of documents.

        :param documents: A list of strings, where each string is a document.
        """
        self.documents = documents

    def retrieve(self, query):
        """
        Retrieves the document most relevant to the given query.

        :param query: A string representing the user's query.
        :return: The document that best matches the query.
        """
        # Lowercase the query for case-insensitive matching
        query = query.lower()
        best_doc = None
        max_overlap = 0

        # Iterate through documents to find the best match
        for doc in self.documents:
            # Count the number of query words present in the document
            overlap = sum(query_word in doc.lower() for query_word in query.split())

            # Update the best match if this document has more overlap
            if overlap > max_overlap:
                best_doc = doc
                max_overlap = overlap

        return best_doc if best_doc else "No relevant document found."


# Example usage
if __name__ == "__main__":
    documents = [
        "Python is a high-level programming language.",
        "Machine learning is a field of artificial intelligence.",
        "The Pacific Ocean is the largest ocean on Earth."
    ]

    retriever = SimpleRetriever(documents)

    query = "Tell me about Python."
    print(f"Query: {query}\nRetrieved: {retriever.retrieve(query)}")
