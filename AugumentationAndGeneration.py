## importing important functions to use during augumentation and generation
from retriever import reranking,reciprocal_rank_fusion,keyword_search,semantic_search

## This function retrieves top_k documents using(semantic_search,keyword_search,rrf and reranking)
## Auguments retrieved_documents with current_query
def augument(query,
             hyde_model,
             tokenizer,
             embedding_model,
             content,
             collection,
             reranker,
             top_k_semantic_and_keyword=50,
             top_k_from_combined_semantic_and_keyword=20,
             final_top_k=5,
             K=60,
             is_generate_summary=False):
    ## get semantic_result
    semantic_results=semantic_search(hyde_model,tokenizer,embedding_model,query, content, collection, top_k=top_k_semantic_and_keyword,is_generate_summary=is_generate_summary)
    ## get keyword_search
    keyword_results=keyword_search(query, collection, top_k=top_k_semantic_and_keyword)
    ## rank those document
    rrf_results=reciprocal_rank_fusion(
        semantic_results,
        keyword_results,
        k=K,
        top_k=top_k_from_combined_semantic_and_keyword
    )
    ## rerank documetn using reranker(cross_encoder)
    top_k_documents=reranking(query,rrf_results,final_top_k,reranker)

    context_parts = []
    ## retrieve most important info from top_k documents like source_file,page and text 
    for i, doc in enumerate(top_k_documents, start=1):

        context_parts.append(
            f"""
Document {i}
Source: {doc["source_file"]}
Page: {doc["page"]}

{doc["text"]}
"""
        )
    ## since tokenizer expects plain text join the content of context_parts into sinle long text
    context = "\n".join(context_parts)

    ## final prompt: augument context, current_query and create a prompt to make model behave like an educatinal assistant
    prompt = f"""
You are an educational assistant for MIT 6.006.

Answer the student's question using the retrieved lecture
context below.

Rules:
- Use the retrieved context as the primary source.
- Do not invent information that is not supported by the context.
- Explain the concept clearly and step-by-step.
- If the context does not contain enough information to answer,
  say that the available lecture context is insufficient.

Retrieved context:
{context}

Student question:
{query}

Answer:
"""

    return prompt
##  This function takes the final augumented prompt, puts into hugging_face model and generate final text 
def generate(query,
             model,
             tokenizer,
             hyde_model,
             hyde_tokenizer,
             embedding_model,
             content,
             collection,
             reranker,
             top_k_semantic_and_keyword=50,
             top_k_from_combined_semantic_and_keyword=20,
             final_top_k=5,
             K=60,
             is_generate_summary=False):

    ## get augumented prompt
    augumented_prompt=augument(query,
            hyde_model,
            hyde_tokenizer,
            embedding_model,
             content,
             collection,
             reranker,
             top_k_semantic_and_keyword= top_k_semantic_and_keyword,
             top_k_from_combined_semantic_and_keyword=top_k_from_combined_semantic_and_keyword,
             final_top_k=final_top_k,
             K=K,
             is_generate_summary=is_generate_summary)
    ## tokenize augumetned_prompt using hugging_face pretrained_tokenizer
    messages = [
        {
            "role": "system",
            "content": (
                "You are an educational assistant for MIT 6.006. "
                "Answer clearly in plain markdown. Do not use \\boxed{} or LaTeX "
                "delimiters. Do not repeat yourself. Finish with a complete answer, then stop."
            ),
        },
        {"role": "user", "content": augumented_prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=700,
        do_sample=False,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
    )

    input_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    ).strip()

    return response