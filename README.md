# Client_Side_EasyRead_NDIS
Client side easy read solution using fine tuned LLMs.

At Intelife and NDIS, we often encounter clients with various kinds of intellectual disabilities. Naturally, comprehension of textual information often presents as a massive challenge to them.

Easy Read is a way of writing to present information so that it is easier for people with low literacy to read. Easy Read documents combine text with layout and imagery to simplify and explain information. However, the process of converting a plain-language document to Easy Read format is an incredibly taxing and time-consuming process, which in turn severely limits the scope of information which the client can receive.

In this project at Intelife Group, I will attempt to utilize fine-tuning using a corpus of existing plain text to Easy Read documents to train an LLM to automate the conversion process, saving a significant amount of human resources and time. This project aims to produce a MVP which can be easily operated by a client without the need for suprvision or intervention by case workers.

The difficulty of converting a document to Easy Read format, which is present in both manual conversion and potentially language model-based conversion, is that a breakpoint between paragraphs or sentences must be selected to summarize the content in between each breakpoint. This is important as it will provide enough context to facilitate an "inter-paragraph" understanding in the client, and enables the summary of each section to be as concise and clear as possible.
