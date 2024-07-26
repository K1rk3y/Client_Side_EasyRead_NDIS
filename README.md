# Client_Side_EasyRead_NDIS
Client side easy read solution using fine tuned LLMs.

At Intelife and NDIS, we often encounter clients with various kinds of intellectual disabilities. Natrually, comprehension of textual infomation often presents as a massive challange to them. 

Easy read is a way of writing to present information so that it is easier for people with low literacy to read. Easy Read documents combine text with layout and imagery to simplify and explain information. However, the process of converting a plain-language document to easy read format is incredibly taxing and time consuming process, which in turn severely limits the scope of infomation which the clinet can recieve. 

In this project at Intelife Group, I will attempt to utlise fine tuning using a corpus of existing plain text to easy read documents to train a LLM to automate the conversion process, svaing significant amount of human resources and time.

The difficulty of converting a document to easy read format which is present in both manual conversion and potentially language model based conversion, is that a breakpoint between paragraphs or sentences must be selected to summerise the content inbetween each breakpoints. This is important as it will provide enough context to facilitate a "inter-paragraph" understanding in the client, and enables the summary of each section to be as concise and clear as possible.
