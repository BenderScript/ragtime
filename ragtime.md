
# Course: Introduction to Retrieval-Augmented Generation (RAG) using Langchain

## Course Description
This course introduces Retrieval-Augmented Generation (RAG) using Langchain, blending theoretical insights with practical coding examples. Emphasis is placed on integrating OpenAI's language models for RAG applications.

## Modules

### Module 1: Introduction to Generative AI

#### Theory: Overview of Generative AI Models and Their Evolution

**1. Understanding Generative AI**:
   - **Definition**: Exploring what generative AI is, including its ability to create new content and make predictions based on learned data patterns.
   - **Key Concepts**: Discussing concepts such as supervised vs. unsupervised learning, neural networks, and deep learning as they pertain to generative AI.

**2. Historical Context and Evolution**:
   - **Early Beginnings**: Tracing the origins of generative AI, from basic neural networks to more complex architectures.
   - **Milestones**: Reviewing key developments like GANs (Generative Adversarial Networks), transformer models, and breakthroughs like GPT (Generative Pre-trained Transformer) series and BERT (Bidirectional Encoder Representations from Transformers).

**3. Types of Generative AI Models**:
   - **Varieties and Uses**: Exploring different types of generative models, including GANs, VAEs (Variational Autoencoders), and language models like GPT and BERT.
   - **Applications**: Discussing how these models are used in various fields such as art, music, text generation, and more.

**4. Ethical Considerations**:
   - **Responsibility**: Understanding the ethical implications of generative AI, including concerns about bias, misinformation, and the impact on digital content authenticity.

#### Practical: Simple Text Generation using OpenAI's Model

**Objective**: Introduce students to practical usage of generative AI with a focus on text generation using OpenAI's GPT model.

**Environment Setup**:
- Python environment.
- Langchain installed.
- Access to OpenAI API.

**Example: Generating Text with OpenAI's GPT Model**

```python
from dotenv import load_dotenv
import os
from langchain.llms import OpenAI


def generate_text(prompt, model):
    # Generate a response using the language model
    return model.generate(prompts=[prompt], max_tokens=4000)


def main():
    # Load environment variables
    load_dotenv()

    # Retrieve the API key
    api_key = os.getenv("OPENAI_API_KEY")

    # Initialize OpenAI's language model with the API key
    llm = OpenAI(api_key=api_key)

    # Example text generation
    prompt = "Write a short story about Bender from Futurama in first person using his language and mannerisms"
    print("Generated Text:", generate_text(prompt, llm))


main()
```

In this module, students will:
- Gain an understanding of generative AI, its history, and the different types of models.
- Learn about the practical applications and ethical considerations of generative AI.
- Get hands-on experience with text generation using OpenAI's GPT model.

This module sets the foundation for understanding generative AI, paving the way for more advanced topics and practical implementations in subsequent modules.

---

### Module 2: Understanding Retrieval in AI
#### Theory: Explanation of Information Retrieval, Its Importance in AI, and Overview of Retrieval Systems

##### Understanding Information Retrieval

Information Retrieval (IR) is the science of searching for information in documents, searching for documents themselves, and also searching for metadata that describes data, and for databases of texts, images, or sounds. It involves the retrieval of information that is relevant to specific user queries from large collections of data, like databases, websites, or document repositories.

#### Importance of Information Retrieval in AI

1. **Foundation for Many AI Applications**: IR is fundamental in various AI applications like search engines, recommendation systems, and data mining. These systems rely on efficient IR to provide relevant, accurate, and timely information.

2. **Enhancing User Experience**: In the context of AI, IR systems are designed to understand and interpret user queries, often in natural language, and fetch the most relevant information. This capability is crucial for enhancing user interaction and experience with AI systems.

3. **Data Organization and Accessibility**: IR plays a vital role in organizing vast amounts of data and making it accessible. It enables AI systems to sift through large datasets to find meaningful patterns, trends, or specific information.

4. **Supporting Decision Making**: In business and research, IR aids in decision-making by providing relevant data and insights. For instance, AI-driven market analysis tools use IR to gather and interpret market data for strategic planning.

5. **Natural Language Processing (NLP) Integration**: IR techniques are closely linked with NLP, as understanding and processing human language is crucial for effective retrieval. This integration is key in chatbots, virtual assistants, and other AI applications involving language understanding.

#### Overview of Retrieval Systems

1. **Search Engines**: The most common form of IR systems. They index and retrieve web pages based on user queries. Examples include Google, Bing, and Yahoo.

2. **Database Management Systems (DBMS)**: These systems retrieve data from structured databases. They use SQL (Structured Query Language) for querying and retrieving data.

3. **Document Retrieval Systems**: These are specialized in retrieving parts of documents, entire documents, or collections of documents. Libraries and academic databases are examples.

4. **Multimedia Retrieval Systems**: They focus on retrieving various forms of multimedia content like images, videos, and audio. They often use metadata or content-based retrieval methods.

5. **Enterprise Search Systems**: Used in businesses to search for information within the organization. They index internal documents, emails, and other data sources.

6. **Recommendation Systems**: Though primarily for suggesting items to users, they rely heavily on IR to filter and present items (like books, movies, products) relevant to a user's interests.

Each of these systems uses different algorithms and techniques tailored to their specific type of data and user needs. The effectiveness of an IR system is often measured by its relevance (the accuracy of the results to the query) and recall (the ability of the system to retrieve all relevant items).#### Practical
- Simple Python script for basic information retrieval.

```python
# Simple Python script for basic information retrieval
# Here, we use a fixed list of documents for simplicity

documents = [
    "The sky is blue.",
    "Apples are sweet.",
    "The ocean is vast."
]

query = "What is the color of the sky?"

# Simple retrieval function
def retrieve_info(query_param, doc_list):
    for doc in doc_list:
        if query_param.lower() in doc.lower():
            return doc
    return "No relevant document found."

def main():
    result = retrieve_info(query, documents)
    print(result)

# Call main function directly
main()
```

---

### Module 3 : Detailed Explanation of RAG, Its Components, and How It Combines Retrieval and Generation

#### What is Retrieval-Augmented Generation (RAG)?

Retrieval-Augmented Generation (RAG) is a hybrid AI model that combines two key components of AI: retrieval (information search and extraction) and generation (creating coherent text based on the retrieved information). RAG models are designed to improve the quality and relevance of generated text by first consulting a vast repository of information (like documents or databases) before producing an answer.

#### Components of RAG

1. **Retrieval System**:
   - **Function**: Searches through a large dataset or document repository to find information relevant to a given query.
   - **Techniques Used**: Often involves indexing techniques, search algorithms, and sometimes machine learning models to understand the query and retrieve pertinent data.

2. **Generative Model**:
   - **Function**: Takes the retrieved information and uses it to generate a coherent and contextually relevant response.
   - **Common Models Used**: Language models like GPT-3, which can generate natural language text based on the input they receive.

#### How RAG Combines Retrieval and Generation

1. **Query Processing**: The process begins with a user query or prompt.
2. **Retrieval Phase**: The retrieval system searches its dataset to find information related to the query. This can include documents, snippets, or structured data.
3. **Integration**: The retrieved information is then fed into the generative model, along with the original query.
4. **Generation Phase**: The generative model uses both the query and the retrieved information to produce a detailed, informed response. This response is typically more accurate and relevant than what a standalone generative model could produce.

#### Diagrams Illustrating RAG

To aid in visualizing the RAG process, I will create two diagrams:

1. **Overall Flow of RAG**: This diagram will show the flow from query to response, highlighting the retrieval and generation phases.
2. **RAG System Architecture**: This diagram will depict the internal components of a RAG model, showcasing how retrieval and generative models are interconnected.

Let's generate these diagrams.

#### Diagram 1: Overall Flow of RAG
- Description: A flowchart starting with the 'User Query' at the left, leading to 'Retrieval System' which searches a 'Data Repository', followed by 'Generative Model' which combines the query and retrieved data to produce the 'Response'.

#### Diagram 2: RAG System Architecture
- Description: A system architecture diagram. At the center is the 'RAG Model'. It is connected on the left to 'User Query' and 'Data Repository', and on the right to the 'Generated Response'. Inside the RAG Model, two interconnected components are shown: 'Retrieval System' and 'Generative Model (e.g., GPT-3)'.

---

### Module 4: Basics of Langchain

#### Theory: Understanding the Langchain Framework

**What is Langchain?**
- Langchain is a Python framework designed to facilitate the integration and utilization of large language models (LLMs) in various applications. It provides tools and abstractions that make it easier to build applications that leverage language models for tasks like text generation, information retrieval, and more.

**Key Features of Langchain:**
1. **Modular Design**: Langchain's architecture is built on modular components, allowing users to mix and match different elements like language models, retrieval systems, and more.
2. **Support for Various Models**: It supports a range of language models, including OpenAI's GPT models, allowing for flexibility in choosing the appropriate model for specific tasks.
3. **Integration with Retrieval Systems**: Langchain facilitates the combination of language models with retrieval systems, enabling the development of sophisticated applications like RAG.
4. **Customization and Extensibility**: The framework is designed for customization, letting developers extend or modify components to fit their specific needs.
5. **Simplifying Complex Workflows**: Langchain streamlines the process of combining different AI and NLP techniques, making it easier to build complex language processing workflows.

#### Practical: Basic Operations with Langchain

**Objective**: To familiarize students with Langchain's basic operations, focusing on initializing language models and performing simple text generation tasks.

**Environment Setup**: 
- Ensure Python is installed.
- Install Langchain (`pip install langchain`).
- Set up OpenAI API keys if not already done.

**Code Example**:

```python
from dotenv import load_dotenv
import os
from langchain.llms import OpenAI

# Load environment variables
load_dotenv()

# Retrieve the API key
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI's language model with the API key
llm = OpenAI(api_key=api_key)

def generate_text(language_model, prompt):
    # Simple text generation
    response = language_model.generate(prompts=[prompt])
    return response

def main():
    # Prompt for text generation
    prompt = "Describe the benefits of using the Langchain framework."
    response = generate_text(llm, prompt)

    # Output the response
    print("Generated Text:", response)

main()

```
---

### Module 5: Implementing Retrieval with Langchain
#### Theory
- Deep dive into retrieval mechanics within Langchain.
#### Practical
- Building a retrieval system using Langchain's `LocalRetriever`.

```python
from dotenv import load_dotenv
import os
from langchain.llms import OpenAI

# Load environment variables
load_dotenv()

# Retrieve the API key
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI's language model with the API key
llm = OpenAI(api_key=api_key)

# Sample documents for a basic retrieval system
documents = {
    "python programming": "Python is a high-level, interpreted programming language known for its readability.",
    "langchain framework": "Langchain is a framework that facilitates the use of language models in various applications."
}

# Simple retrieval function
def retrieve(query, doc_dict):
    for key, value in doc_dict.items():
        if key in query.lower():
            return value
    return "Relevant information not found."

# Function to generate a response using retrieved information
def generate_response_with_retrieval(prompt, doc_dict, language_model):
    # Retrieve relevant information based on the prompt
    retrieved_info = retrieve(prompt, doc_dict)
    
    # Combine the prompt with the retrieved information
    combined_input = f"{prompt}\n\nRetrieved Info: {retrieved_info}"
    
    # Generate response using the language model
    return language_model.generate(prompts=[combined_input])


def main():
# Generate a response
    prompt = "Tell me about the Langchain framework."
    response = generate_response_with_retrieval(prompt, documents, llm)

    print("Response with Retrieved Information:", response)

main()


```

---

### Module 6: Integrating RAG in Langchain
#### Expanded Theory for Module 6: How Langchain Facilitates the Integration of Retrieval and Generation

**Overview of Integration in Langchain**
- Langchain, as a Python framework, is designed to simplify the integration of different components required for advanced AI applications. Specifically, it focuses on combining retrieval systems with generative language models, facilitating the creation of sophisticated systems like Retrieval-Augmented Generation (RAG).

**1. Modular Approach**:
   - **Explanation**: Langchain uses a modular architecture, where each component (like retrieval systems and language models) can be independently developed, tested, and integrated. This modularity allows for flexibility and customization in building RAG systems.
   - **Benefits**: Easier experimentation with different configurations, leading to more tailored and efficient solutions.

**2. Support for Various Language Models**:
   - **Integration**: Langchain supports various language models, including those from OpenAI. This allows developers to choose the most suitable model for their application's needs.
   - **Advantages**: Flexibility in model selection based on factors like model size, performance, and cost.

**3. Customizable Retrieval Mechanisms**:
   - **Functionality**: Langchain provides the ability to integrate custom retrieval mechanisms, whether they are simple keyword-based systems or more complex machine learning-driven solutions.
   - **Use Case**: This is crucial for applications where the retrieval needs are specific to a particular domain or type of data.

**4. Seamless Combination of Components**:
   - **Workflow**: Langchain streamlines the process of combining retrieval results with generative model inputs. It manages the intricacies of feeding retrieved information into the language model to generate coherent and contextually relevant responses.
   - **Implication**: Reduces the technical barrier for developers, making it easier to create RAG systems without deep expertise in both retrieval and generation technologies.

**5. Handling Complex Queries**:
   - **Capability**: By integrating retrieval with generation, Langchain can handle more complex queries that require external knowledge or context not contained within the language model itself.
   - **Impact**: Enhances the quality and relevance of the responses generated by the system.

**6. Extensibility and Scalability**:
   - **Design Philosophy**: Langchain is designed to be extensible and scalable. Developers can start with simple implementations and gradually scale up to more complex systems as needed.
   - **Advantage**: Suitable for both small-scale experiments and large-scale production applications.

**7. Community and Ecosystem**:
   - **Support**: Langchain's growing community and ecosystem offer resources, examples, and support, which can be invaluable for developers working on RAG systems.
   - **Benefit**: Access to shared knowledge and solutions helps accelerate development and problem-solving.

In summary, Langchain's design and features make it an ideal framework for developers looking to explore and implement RAG systems. Its modular architecture, support for various language models and retrieval mechanisms, and focus on ease of integration, scalability, and community support, provide a robust foundation for building sophisticated AI applications.#### Practical
- Implementing a basic RAG model using Langchain.

```python
from dotenv import load_dotenv
import os
from langchain.llms import OpenAI

# Load environment variables
load_dotenv()

# Retrieve the API key
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI's language model with the API key
llm = OpenAI(api_key=api_key)

# Sample documents for a basic retrieval system
documents = {
    "machine learning": "Machine learning is a type of artificial intelligence that allows software applications to become more accurate at predicting outcomes without being explicitly programmed to do so.",
    "python programming": "Python is a popular programming language for machine learning due to its simplicity and versatility."
}

# Simple retrieval function
def retrieve(query, doc_dict):
    for key, value in doc_dict.items():
        if key in query.lower():
            return value
    return "Relevant information not found."

# Function to generate a response using retrieved information
def generate_response_with_retrieval(prompt, doc_dict, language_model):
    # Retrieve relevant information based on the prompt
    retrieved_info = retrieve(prompt, doc_dict)
    
    # Combine the prompt with the retrieved information
    combined_input = f"{prompt}\n\nAdditional Info: {retrieved_info}"
    
    # Generate response using the language model
    return language_model.generate(prompts=[combined_input])

def main():
    # Generate a response
    prompt = "Explain machine learning and its relation to Python."
    response = generate_response_with_retrieval(prompt, documents, llm)

    print("Response with Retrieved Information:", response)


main()



```
---

### Module 7: Advanced RAG Techniques in Langchain

#### Theory: Exploring Advanced Features of RAG

**1. Customizing Retrieval Sources**:
   - **Explanation**: Learn how to tailor the retrieval component in RAG to use various data sources like custom databases, web scrapings, or specialized corpora.
   - **Importance**: This enables the RAG model to pull information from sources that are most relevant to your specific application needs.

**2. Fine-Tuning Language Models**:
   - **Explanation**: Dive into methods for fine-tuning language models like GPT-3 to better suit the specific context or domain of your application.
   - **Benefits**: Enhances the relevance and accuracy of the generative component in RAG.

**3. Combining Multiple Retrieval Systems**:
   - **Explanation**: Understand how to integrate multiple retrieval systems within a single RAG setup to broaden the scope of information sourcing.
   - **Advantage**: This approach can significantly improve the model's ability to generate comprehensive and nuanced responses.

**4. Optimizing Performance**:
   - **Focus**: Strategies for optimizing the efficiency and speed of RAG models, especially in resource-intensive applications.
   - **Relevance**: Essential for deploying RAG models in production environments where performance and resource utilization are critical.

#### Practical: Implementing Advanced RAG Applications

**Objective**: To provide hands-on experience with advanced RAG techniques, focusing on customization and optimization using Langchain and OpenAI models.

**Environment Setup**:
- Python environment with Langchain installed.
- Access to OpenAI API.

**Example 1: Customizing Retrieval Sources**

```python
from dotenv import load_dotenv
import os
from langchain.llms import OpenAI

# Load environment variables
load_dotenv()

# Retrieve the API key
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI's language model with the API key
llm = OpenAI(api_key=api_key)

# Sample documents for custom retrieval logic
documents = {
    "quantum computing": "Quantum computing harnesses quantum mechanics to process information in novel ways.",
    "cybersecurity": "Cybersecurity involves protecting systems and networks from digital attacks."
}

# Basic custom retrieval function
def custom_retrieve(query, doc_dict):
    for key, value in doc_dict.items():
        if key in query.lower():
            return value
    return "Specific information not found."

# Function to generate a response with additional context
def generate_response_with_custom_context(prompt, doc_dict, language_model):
    # Retrieve context based on the prompt
    context = custom_retrieve(prompt, doc_dict)
    
    # Combine the prompt with the retrieved context
    combined_input = f"{prompt}\n\nContext: {context}"
    
    # Generate response using the language model
    return language_model.generate(prompts=[combined_input])

def main():
    # Generate a response
    prompt = "What is the impact of quantum computing on cybersecurity?"
    response = generate_response_with_custom_context(prompt, documents, llm)

    print("Response with Custom Context:", response)


main()

```

**Example 2: Integrating Multiple Retrieval Systems**

```python
from dotenv import load_dotenv
import os
from langchain.llms import OpenAI

# Load environment variables
load_dotenv()

# Retrieve the API key
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI's language model with the API key
llm = OpenAI(api_key=api_key)

# Sample documents for basic retrieval logic
documents = {
    "renewable energy": "Renewable energy is energy from sources that are naturally replenishing.",
    "solar power": "Solar power is energy from the sun that is converted into thermal or electrical energy."
}

# Basic retrieval function
def retrieve(query, doc_dict):
    for key, value in doc_dict.items():
        if key in query.lower():
            return value
    return "Information not found."

# Function to enhance response with retrieved information
def generate_response_with_context(prompt, doc_dict, language_model):
    # Retrieve context based on the prompt
    context = retrieve(prompt, doc_dict)
    
    # Generate response using the language model
    combined_input = f"{prompt}\n\nContext: {context}"
    return language_model.generate(prompts=[combined_input])


def main():
    # Generate a response
    prompt = "Tell me about solar power and its benefits."
    response = generate_response_with_context(prompt, documents, llm)

    print("Response with Context:", response)


main()


```

---

### Module 8: Real-World Applications and Case Studies of RAG

#### Theory: Understanding Real-World Applications of RAG

1. **Search Engines and Information Retrieval**:
   - Application of RAG models to enhance the relevance and accuracy of search engine results. RAG can be used to augment traditional keyword-based searches with contextually enriched responses.

2. **Customer Service Chatbots**:
   - Implementing RAG in chatbots allows for more accurate and context-aware responses, improving customer engagement and satisfaction.

3. **Content Creation and Summarization**:
   - RAG models can assist in generating new content or summarizing existing content by retrieving relevant information from a large corpus and integrating it into coherent narratives or summaries.

4. **Question Answering Systems**:
   - RAG systems are particularly effective in complex question-answering scenarios where the answer requires external knowledge not contained within the model itself.

5. **Educational Tools and Research**:
   - Utilizing RAG for educational purposes, such as creating study guides or research aids that can pull in relevant information from a wide array of sources.

#### Practical: Building a Simple RAG-Like System for Content Summarization

**Objective**: To create a simplified RAG-like system using Langchain and OpenAI's models for the purpose of content summarization.

**Environment Setup**:
- Python environment with Langchain installed.
- Access to OpenAI API.

**Example: Simple Content Summarization System**

```python
from dotenv import load_dotenv
import os
from langchain.llms import OpenAI

# Load environment variables
load_dotenv()

# Retrieve the API key
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI's language model with the API key
llm = OpenAI(api_key=api_key)

# A basic retrieval function (simulating a simplistic retrieval system)
def retrieve_content(topic, content_db):
    for content in content_db:
        if topic.lower() in content.lower():
            return content
    return "Content not found."

# Function to summarize content on a given topic
def summarize_content(topic, content_database, language_model):
    # Retrieve related content
    content = retrieve_content(topic, content_database)
    
    # Check if content was found
    if content == "Content not found.":
        return content

    # Generate a summary using the language model
    summary_prompt = f"Summarize the following content: {content}"
    return language_model.generate(prompts=[summary_prompt])

def main():
    # Sample content database
    content_db = [
        "Python is a high-level, interpreted programming language with dynamic semantics.",
        "Machine learning is an application of artificial intelligence that provides systems the ability to automatically learn."
    ]

    # Generate a summary
    topic = "Python"
    summary = summarize_content(topic, content_db, llm)

    print("Content Summary:", summary)


main()

```

---

### Module 9: Best Practices and Optimization for RAG Systems

#### Theory: Enhancing Performance and Best Practices

1. **Performance Optimization**:
   - Techniques to improve the efficiency of RAG systems, such as optimizing retrieval algorithms, caching frequently accessed data, and fine-tuning language models for specific tasks.
   
2. **Handling Large Data Sets**:
   - Strategies for dealing with large-scale data, including efficient data indexing, using distributed computing, and employing data compression techniques.

3. **Quality of Responses**:
   - Ensuring the quality of responses generated by RAG systems through techniques like response filtering, validation checks, and incorporating feedback loops.

4. **Scalability Considerations**:
   - Approaches to ensure RAG systems can scale effectively, including modular architecture design, load balancing, and resource management.

5. **Ethical Considerations and Bias Mitigation**:
   - Addressing ethical considerations and mitigating biases in RAG systems, focusing on fairness, transparency, and responsible AI practices.

#### Practical: Implementing Optimization Techniques in a RAG-Like System

**Objective**: To demonstrate basic optimization techniques for a simplified RAG-like system using Langchain and OpenAI's models.

**Environment Setup**:
- Python environment with Langchain installed.
- Access to OpenAI API.

**Example: Optimized Content Retrieval and Generation System**

```python
from dotenv import load_dotenv
import os
from langchain.llms import OpenAI

# Load environment variables
load_dotenv()

# Retrieve the API key
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI's language model with the API key
llm = OpenAI(api_key=api_key)


class CachedRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.cache = {}  # Cache to store retrieval results

    def retrieve(self, query):
        if query in self.cache:
            return self.cache[query]

        for doc in self.documents:
            if query.lower() in doc.lower():
                self.cache[query] = doc
                return doc

        self.cache[query] = "Content not found."
        return "Content not found."


def generate_response(topic, retriever, llm):
    content = retriever.retrieve(topic)

    if content == "Content not found.":
        return content

    response_prompt = f"Discuss the following topic: {content}"
    # Adjusted to use a hypothetical 'generate' method
    return llm.generate(prompts=[response_prompt], max_tokens=50)


def main():
    content_db = [
        "Python is a widely-used programming language.",
        "Machine learning enables computers to learn from data."
    ]

    cached_retriever = CachedRetriever(content_db)

    topic = "Python"
    response = generate_response(topic, cached_retriever, llm)

    print("Generated Response:", response)


main()
```

---

### Module 10: Course Project - Building a RAG-Based Application

#### Project Overview

**Objective**: To apply the concepts learned throughout the course by building a RAG-based application using Langchain and OpenAI's models. The project will involve creating a simple question-answering system that retrieves information from a set of documents and generates responses to user queries.

**Scope**:
- Students will design and implement a basic RAG system.
- The system will use a simple retrieval method combined with OpenAI's language models for generating responses.
- The project focuses on integrating retrieval and generation processes, demonstrating understanding and practical application of RAG concepts.

#### Practical: Building a Simple Question-Answering System

**Environment Setup**:
- Python environment with Langchain installed.
- Access to OpenAI API.

**Example: Simple Question-Answering System Using RAG Principles**

```python
from dotenv import load_dotenv
import os
from langchain.llms import OpenAI

def main():
    # Load environment variables
    load_dotenv()

    # Retrieve the API key
    api_key = os.getenv("OPENAI_API_KEY")

    # Initialize OpenAI's language model with the API key
    llm = OpenAI(api_key=api_key)

    # Sample documents database
    documents = [
        "Python is a high-level, interpreted programming language.",
        "Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data."
    ]

    # Initialize the retriever
    retriever = DocumentRetriever(documents)

    # Example query
    query = "What is machine learning?"
    response = answer_query(query, retriever, llm)

    print("Response:", response)

class DocumentRetriever:
    def __init__(self, document_list):
        self.documents = document_list

    def retrieve(self, query):
        # Simple keyword-based retrieval
        for doc in self.documents:
            if query.lower() in doc.lower():
                return doc
        return "No relevant document found."

def answer_query(query, retriever, language_model):
    # Retrieve relevant document
    document = retriever.retrieve(query)

    # If no document is found, return a default response
    if document == "No relevant document found.":
        return "I'm sorry, I don't have information on that topic."

    # Use the language model to generate an answer
    answer_prompt = f"Answer the question based on the following information: {document}\n\nQuestion: {query}"
    return language_model.generate(prompts=[answer_prompt], max_tokens=150)


main()

```
