from retriever import reranking,reciprocal_rank_fusion,keyword_search,semantic_search

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
    semantic_results=semantic_search(hyde_model,tokenizer,embedding_model,query, content, collection, top_k=top_k_semantic_and_keyword,is_generate_summary=is_generate_summary)
    keyword_results=keyword_search(query, collection, top_k=top_k_semantic_and_keyword)
    rrf_results=reciprocal_rank_fusion(
        semantic_results,
        keyword_results,
        k=K,
        top_k=top_k_from_combined_semantic_and_keyword
    )
    top_k_documents=reranking(query,rrf_results,final_top_k,reranker)

    context_parts = []

    for i, doc in enumerate(top_k_documents, start=1):

        context_parts.append(
            f"""
Document {i}
Source: {doc["source_file"]}
Page: {doc["page"]}

{doc["text"]}
"""
        )
    context = "\n".join(context_parts)

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
    
    inputs = tokenizer(
            augumented_prompt,
            return_tensors="pt"
        ).to(model.device)
    
    outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=False
        )
    
    input_length = inputs["input_ids"].shape[1]
    
    response = tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        ).strip()
    
    return response



