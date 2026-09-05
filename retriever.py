
def generate_summary(hyde_model,tokenizer,content):
    formatted = "\n".join(f'{m["role"]}: {m["content"]}' for m in content)
    summary_prompt = f"""
Summarize the following conversation for use as context in a
future question-answering system.

Keep:
- topics the user is learning
- concepts already discussed
- important questions asked
- important context needed to understand the user's next question

Remove:
- greetings
- repetition
- irrelevant conversation
- unnecessary details

Conversation:
{formatted}

Conversation summary:
"""

    inputs = tokenizer(
        summary_prompt,
        return_tensors="pt"
    ).to(hyde_model.device)

    outputs = hyde_model.generate(
        **inputs,
        max_new_tokens=250,
        do_sample=False
    )

    input_length = inputs["input_ids"].shape[1]

    summary = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    ).strip()

    return summary
def generate_query_embedding_for_semantic_search(hyde_model,tokenizer,embedding_model,current_query,previous_chat,is_generate_summary=False):

    if (is_generate_summary==True):
        generated_summary=generate_summary(hyde_model,tokenizer,previous_chat)
    else:
         generated_summary=""
    prompt = f"""
Write a hypothetical passage that would answer the following question.
The passage should contain the technical information that would likely
appear in a relevant MIT 6.006 lecture.

Question:
{current_query}
Summary:
{generated_summary}

Hypothetical passage:
"""
    inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(hyde_model.device)
    hyde_outputs = hyde_model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True
)
    input_length = inputs["input_ids"].shape[1]

    generated_tokens = hyde_outputs[0][input_length:]

    document = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)
    query_embedding=embedding_model.encode(document,normalize_embeddings=True)
    return query_embedding

def semantic_search(hyde_model,tokenizer,embedding_model,current_query, previous_chat, collection, top_k=50,is_generate_summary=False):

    query_embedding = generate_query_embedding_for_semantic_search(
         hyde_model,
         tokenizer,
         embedding_model,
        current_query,
        previous_chat,
        is_generate_summary
    )

    results = collection.query.near_vector(
        near_vector=query_embedding.tolist(),
        limit=top_k
    )

    return results.objects

def keyword_search(current_query, collection, top_k=50):
    results = collection.query.bm25(
        query=current_query,
        limit=top_k
    )
    return results.objects

def reciprocal_rank_fusion(
    semantic_results,
    keyword_results,
    k=60,
    top_k=20
):
    
    rrf_scores = {}
    documents = {}
    semantic_rank={}
    keyword_rank={}

    # Semantic search ranking
    for rank, obj in enumerate(semantic_results, start=1):

        chunk_id = obj.properties["chunk_id"]

        score = 1 / (k + rank)

        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0) + score
        )

        documents[chunk_id] = obj
        semantic_rank[chunk_id]=rank


    # Keyword/BM25 ranking
    for rank, obj in enumerate(keyword_results, start=1):

        chunk_id = obj.properties["chunk_id"]

        score = 1 / (k + rank)

        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0) + score
        )

        documents[chunk_id] = obj
        keyword_rank[chunk_id]=rank


    # Sort by RRF score
    ranked = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


    # Return top-k documents
    final_results = []

    for chunk_id, score in ranked[:top_k]:

        obj = documents[chunk_id]
        final_result={
            "chunk_id": chunk_id,
            "text": obj.properties["text"],
            "source_file": obj.properties["source_file"],
            "page": obj.properties["page"],
            "course": obj.properties["course"],
            "rrf_score": score
        }
        if chunk_id in semantic_rank.keys():
            final_result['semantic_rank']=semantic_rank[chunk_id]
        else:
            final_result['semantic_rank']=None
        if chunk_id in keyword_rank.keys():
                    final_result['keyword_rank']=keyword_rank[chunk_id]
        else:
                    final_result['keyword_rank']=None
        

        final_results.append(final_result)

    return final_results

## reranking using cross_encoder

def reranking(query,rrf_results,top_k,reranker):
    pairs = [
        (query, doc["text"])
        for doc in rrf_results
    ]
    scores = reranker.predict(pairs)
    scored_documents = []

    for doc, score in zip(rrf_results, scores):

        doc = doc.copy()
        doc["rerank_score"] = float(score)

        scored_documents.append(doc)

    # Sort by cross-encoder score
    scored_documents.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    # Return top-k
    return scored_documents[:top_k]


    
    



    