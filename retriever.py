
## This function is used to summarize previous content which can benefit semantic search since if we don't give context to current query it may not be able to retrieve most relevant documents from collections(not always)

def generate_summary(hyde_model,tokenizer,content):
    ## content is a list in which each element is a dictionary having key as role and value as text
    ## joining role and text to add into summary_prompt since hyde_model expects plain text for tokenizing instead of list
    formatted = "\n".join(f'{m["role"]}: {m["content"]}' for m in content)
    ## augumenting our previous converstion into summary_prompt to summarize previous conversation
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
    ## tokenizing summary_prompt.
    ## hugging face tokenizer converts text into a dictionary of items like input_id and attention_mask
    inputs = tokenizer(
        summary_prompt,
        return_tensors="pt"
    ).to(hyde_model.device)
    ## using our hyde model to convert tokenized summary_prompt into summarized output
    outputs = hyde_model.generate(
        **inputs,
        max_new_tokens=250,## we don't want long summary hence max_new_tokens are set to 250 tokens
        do_sample=False## we want a fix summary hence sampling is set to false
    )

    ## getting length of summary_prompt 
    input_length = inputs["input_ids"].shape[1]
   
    summary = tokenizer.decode(
        outputs[0][input_length:], ## summary will only contain newly generated token
        skip_special_tokens=True ## it skips an special token like <EOS>,<BOS>,<UNK>
    ).strip()

    return summary

## This function uses the generated summary from generate_summary() augument with 
##  current query convert into hypothetical_document_embedding(hyde) using hyde_model 
##  and embedding_model and return query_embedding if we explicitly provides 
##  is_generate_summary=True else we don't augument generated_summary (it saves computation power) 
##  and get query_embedding from current_query
def generate_query_embedding_for_semantic_search(hyde_model,tokenizer,embedding_model,current_query,previous_chat,is_generate_summary=False):
    ## generate_summary only if is_generate_summary is explicitly set to true
    if (is_generate_summary==True):
        generated_summary=generate_summary(hyde_model,tokenizer,previous_chat)
    ## if is_generate_summary==False genrated_summary will be "".
    else:
         generated_summary=""
    ## Creating a prompt from current_query and generated_summary to generate a hypothetical_document(hyde)
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
    ## converting prompt into token ids using hyde_tokenizer
    inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(hyde_model.device)
    ## generating hypothetical_document using hyde_model
    hyde_outputs = hyde_model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=False ## we don't want to generate different hypothetical document everytime hence samling is set False
)
    ## getting input length of prompt
    input_length = inputs["input_ids"].shape[1]
    ## removing input from generated_text i.e hyde_outputs
    generated_tokens = hyde_outputs[0][input_length:]
    ## decoding generated_token into text using tokenizer.decode
    document = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)
    ## embedding document using embedding_model 
    query_embedding=embedding_model.encode(document,normalize_embeddings=True)
    return query_embedding

## This function gets query_embedding using previous function generate_query_embedding_for_semantic_search() and retrieves top_k relevant documents using semantic search(cosine similarity)
def semantic_search(hyde_model,tokenizer,embedding_model,current_query, previous_chat, collection, top_k=50,is_generate_summary=False):
    ## gets query embedding
    query_embedding = generate_query_embedding_for_semantic_search(
         hyde_model,
         tokenizer,
         embedding_model,
        current_query,
        previous_chat,
        is_generate_summary
    )
    ## gets top_k documents using semantic search(cosine_similarity)
    results = collection.query.near_vector(
        near_vector=query_embedding.tolist(),
        limit=top_k
    )
    ## results contains top_k matching documents as objects in it
    ## one result in results contain

#  result
# │
# ├── properties
# │   ├── text
# │   ├── source_file
# │   ├── page
# │   ├── chunk_id
# │   └── course
# │
# ├── uuid
# └── metadata
    ## results are accessed through results.object
    return results.objects

## This function does keyword search(BM-25) to get top_k documents
def keyword_search(current_query, collection, top_k=50):
    ## does keyword search i.e scores chunks based on matching query terms and their importance.
    results = collection.query.bm25(
        query=current_query,
        limit=top_k
    )
    ## results are accessed through results.object
    return results.objects

## Till now we got top_k documents from keyword and semantic search now we will rank those document using a mathematical function.
## Those searches may contain some common document and some different documents.
## It  returns a final_list of top_k documents from total documents of semantic and keyword search
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
        ## get the chunk id from the given chunk
        chunk_id = obj.properties["chunk_id"]
        ## mathematical formula to calculate score
        score = 1 / (k + rank)
        ## if that chunk_id already appeared before add score into it else create new key with value 0 and score into it
        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0) + score
        )
        ## add the object into documents with key as its chunk_id
        documents[chunk_id] = obj
        semantic_rank[chunk_id]=rank


    # Keyword/BM25 ranking
    for rank, obj in enumerate(keyword_results, start=1):
        ## get chunk_id from given chunk using its properties
        chunk_id = obj.properties["chunk_id"]
        ## calculate its score
        score = 1 / (k + rank)
        ## if that chunk_id already appeared before add score into it else create new key with value 0 and score into it
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
        ## storing top_k documents in a list of dictionary having (chunk_id,text,source_file,page,course,rrf_score)
        obj = documents[chunk_id]
        final_result={
            "chunk_id": chunk_id,
            "text": obj.properties["text"],
            "source_file": obj.properties["source_file"],
            "page": obj.properties["page"],
            "course": obj.properties["course"],
            "rrf_score": score
        }
        ## if the document was a part of semantic_search retrieval put its semantic_search rank into dict
        if chunk_id in semantic_rank.keys():
            final_result['semantic_rank']=semantic_rank[chunk_id]
        ## if not a semantic retrieval put semantic_rank=None 
        else:
            final_result['semantic_rank']=None
        ## if the document was a part of Keyword_search retrieval put its keyword_search rank into dict
        if chunk_id in keyword_rank.keys():
                    final_result['keyword_rank']=keyword_rank[chunk_id]
        ## if not a keyword_search retrieval put keyword_search rank=None
        else:
                    final_result['keyword_rank']=None
        
        ## append dict to final result
        final_results.append(final_result)

    return final_results

## reranking chunks using cross_encoder
## This function takes the documents after ranking through rrf(reciprocal_rank_fusion) and rerank them using a cross_encoder.
## Cross_encoder tries to find a semantic relation between query and retrieved documents by 
## appending them together and looking at both pieces together and generates a relevance_score.
## Higher the relevance score more relevant the document is.
def reranking(query,rrf_results,top_k,reranker):
    ## pair query with document text to pass into cross_encder
    pairs = [
        (query, doc["text"])
        for doc in rrf_results
    ]
    ## get the relevance_scores from reranker(cross_encoder)
    scores = reranker.predict(pairs)
    scored_documents = []

    ## get the doc from rrf_results and add score of each doc into the dictionary and append into scored_documents
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


    
    



    