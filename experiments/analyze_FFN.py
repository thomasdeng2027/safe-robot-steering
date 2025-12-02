import torch
import random

from model.smolvla_policy import SmolVLALiberoPolicy

"""
Outputs a list. Contains 1 object, a module that contains the LM head, which has unembedding matrix
Necessary to project value vectors onto the token space
"""
def find_LMhead(policy):
    lm_head = []
    for name, module in policy.named_modules():
        if('lm_head' in name):
            lm_head.append((name, module))
            
    embed_matrix = lm_head[0][1].weight
    
    """
    print(embed_matrix)
    print(embed_matrix.shape)    #Produces a size of 49280, 960. 
    """

    return lm_head

"""
Finding the down projections
The down projection is crucial in the Llama2 architecture, which the SmolLM2 is made of. 
It acts as the parameter matrix that is described in the Haon et al. paper, projecting the hidden layer to the output layer
"""
def find_down_projs(policy):

    downs = []
    for name, module in policy.named_modules():
        if('mlp.down_proj' in name and 'text_model' in name):
            downs.append((name, module))

    #dimensionality is 960 for output, 2560 for hidden layer
    #optionally print a small 5x5 portion of the matrix
    """
    module1 = downs[1][1].weight
    print(module1[:5, :5])
    """

    return downs

"""
Value vector extraction. Layer-specific.
down_projects holds 32 modules/weight_matrices. 2560 value vectors per layer. 32 x 2560 x 960 = 78643200
"""
def extract_value_vectors(down_projs, layer):
    matrix = down_projs[layer][1].weight #gives the weight matrix of the layer
    value_vectors = matrix.T #transpose it.  now indexing gives the value vectors of dim 960

    return value_vectors

"""
Input: value vectors for 1 layer, unembedding matrix
Output: logits (2560, 49280) one probabiltiy distribution for every neuron
"""
def project_tkn_space(value_vectors, embed_mat):
    logits = value_vectors @ embed_mat.T 
    return logits


"""
Input: logits, tokenizer, k.
Output: list of top k tokens for each value vector
"""
def get_top_k_tokens(logits, tokenizer, k):

    #built in pytorch function that returns k largest elements along the token space 
    topk_vals, topk_indx = torch.topk(logits, k, dim=1)
    # both have dim (2560, k) but store the logit values and indices

    topk_tokens = []
    topk_scores = []

    for i in range(len(logits)):
        token_ids = topk_indx[i].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids) #tokenization

        topk_tokens.append(tokens)
        topk_scores.append(topk_vals[i].tolist())

    return topk_tokens, topk_scores


"""
#get the top 30 tokens for each value vector
def top_k_tokens_for_value_vectors(
    logits: torch.Tensor,
    tokenizer,
    k: int = 30,
    max_vectors: int | None = 50,
):
    
    logits: [m, vocab_size]
    returns: list of dicts, one per value vector:
        {
            "index": int,
            "token_ids": [...],
            "tokens": [...],
            "scores": [...]
        }
    
    m, vocab_size = logits.shape
    if max_vectors is not None:
        m = min(m, max_vectors)
        logits = logits[:m]

    topk_vals, topk_ids = torch.topk(logits, k=k, dim=-1)  # [m, k]

    results = []
    for i in range(m):
        ids = topk_ids[i].tolist()
        toks = tokenizer.convert_ids_to_tokens(ids)
        scores = topk_vals[i].tolist()
        results.append({
            "index": i,
            "token_ids": ids,
            "tokens": toks,
            "scores": scores,
        })
    return results

#usage
topk_info = top_k_tokens_for_value_vectors(
    logits,
    tokenizer,
    k=30,
    max_vectors=20,  # just inspect 20 value vectors to start
)

for vec in topk_info:
    print(f"\nValue vector #{vec['index']}")
    for t, s in zip(vec["tokens"], vec["scores"]):
        print(f"  {t:>15s}   {s:.3f}")


"""

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = SmolVLALiberoPolicy(model_name="HuggingFaceVLA/smolvla_libero",device=device)
    #this policy contains the SmolVLA policy, then SmolVLM2, SmolLM2, and the FFNs
    policy.eval()

    down_projs = find_down_projs(policy.policy)
    LM_head = find_LMhead(policy.policy) #still the module
    embed_matrix = LM_head[0][1].weight #convert the module to the tensor

    layer = 30 #as an example
    value_vectors = extract_value_vectors(down_projs, layer)
    logits = project_tkn_space(value_vectors, embed_matrix) 

    k = 31 #as used by Haon et al.
    top_k_tokens, top_k_scores = get_top_k_tokens(logits, policy.tokenizer, k)

    #focusing on 1 token, analyzing its scores and tokens.
    nums = random.sample(range(1, 2561), 10)
    for num in nums:
        print(num)
        print()
        for i in range(len(top_k_tokens[num])):
            if ('Ġ' in top_k_tokens[num][i]):
                top_k_tokens[num][i] = top_k_tokens[num][i].replace("Ġ","")
        print(top_k_tokens[num])


    """
    lib_policy.eval()

    # Underlying LeRobot SmolVLA policy (HuggingFace-style)
    smolvla = lib_policy.policy      # This is SmolVLAPolicy from lerobot

    # 1. find lm_head (unembedding matrix)
    lm_head = find_lm_head(smolvla)

    # 2. find all down_proj layers
    downs = find_down_projs(smolvla)

    # For a first pass, just analyze ONE layer, e.g. the middle one
    name, down_proj = downs[len(downs) // 2]
    print(f"\n[INFO] Analyzing down_proj: {name}")

    # 3. get row-wise value vectors from this down_proj
    value_vectors = get_value_vectors_from_down_proj(down_proj)
    print("[INFO] value_vectors shape:", value_vectors.shape)

    # 4. project into token space with lm_head
    logits = project_value_vectors_to_tokens(value_vectors, lm_head)
    print("[INFO] logits shape:", logits.shape)

    # 5. get top-30 tokens for first N value vectors
    topk = top_k_tokens_for_vectors(
        logits,
        tokenizer=lib_policy.tokenizer,   # wrapper already has the tokenizer
        k=30,
        max_vectors=10,                   # change to inspect more
    )

    for vec in topk:
        print(f"\n==== Value vector #{vec['index']} ====")
        for t, s in zip(vec["tokens"], vec["scores"]):
            print(f"{t:>20s}   {s:.3f}")

"""
if __name__ == "__main__":
    main()