class LocalRetriever:
    def __init__(self, documents):
        """
        Initializes the LocalRetriever with a list of documents.

        :param documents: A list of strings, where each string represents a document.
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
            # Split the query and document into words for comparison
            query_words = set(query.split())
            doc_words = set(doc.lower().split())

            # Calculate the overlap as the size of the intersection set
            overlap = len(query_words.intersection(doc_words))

            # Update the best match if this document has more overlap
            if overlap > max_overlap:
                best_doc = doc
                max_overlap = overlap

        return best_doc if best_doc else "No relevant document found."


# Example usage
if __name__ == "__main__":
    documents = [
        "Python is a versatile programming language used in various fields.",
        "Machine learning involves algorithms and statistical models.",
        "The Atlantic Ocean borders Africa, Europe, and the Americas."
    ]

    retriever = LocalRetriever(documents)

    query = "What can you tell me about Python?"
    print(f"Query: {query}\nRetrieved: {retriever.retrieve(query)}")
