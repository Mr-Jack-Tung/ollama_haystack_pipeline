# Ollama Haystack pipeline

Thấy cái cách haystack pipeline hoạt động là đã muốn test thử xem thế nào rồi. Và kết quả rất Ok nhé ^^

### Trước tiên là cài các packages cần thiết: 
- pip install haystack
- pip install haystack-ai
- pip install ollama-haystack

### Source code:

rag_pipeline = Pipeline()<br>
rag_pipeline.add_component("retriever", retriever)<br>
rag_pipeline.add_component("prompt_builder", prompt_builder)<br>
rag_pipeline.add_component("llm", llm)<br>
rag_pipeline.connect("retriever", "prompt_builder.documents")<br>
rag_pipeline.connect("prompt_builder", "llm")<br>

~> làm theo demo ở đây https://docs.haystack.deepset.ai/docs/get_started , nhưng lại muốn dùng ollama thay cho ChatGPT cơ , nên phải dùng thêm cái này https://docs.haystack.deepset.ai/docs/ollamagenerator . Ok vậy là cũng xong :D

### Projects tham khảo:
- https://github.com/anakin87/autoquizzer : Generates a quiz from a URL. You can play the quiz, or let the LLM play it.
- https://github.com/anakin87/autoquizzer/blob/main/backend/pipelines.py
