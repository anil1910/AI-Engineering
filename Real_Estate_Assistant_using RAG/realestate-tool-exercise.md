from distutils.dep_util import newer

# Problem Statement: Customize the Prompt Template of Real Estate Tool
There are many instances when you'd have to modify the existing behaviour of Langchain to fit to your needs. In this exercise, you will be customizing the prompt template of the Real Estate Tool to align it with the usecase. You can refer to the following discussion to learn how to customize the RetrievalQAWithSourcesChain: [Customizing the Prompt Template](https://github.com/langchain-ai/langchain/issues/3523)
### **Tasks**
1. Create a new `prompt.py` file in the `real-estate-tool` directory. And the relevant code to append the prompt in : `langchain.chains.qa_with_sources.stuff_prompt.template` as shown below:
```python
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources.stuff_prompt import template

new_template = "YOUR PROMPT" + template
PROMPT = PromptTemplate(template=new_template, input_variables=["summaries", "question"])

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)
```
2. Now create the two prompts as shown in the example above.
3. Once you have created the prompts, update the instantiation of `RetrievalQAWithSourcesChain` in the `rag.py` file to use the new prompt template.