"""Inference-only shared-prefix encoding, preserving independent causal paths.

``auto`` packs every path of one state into a token trie. The state is
encoded once for all of its questions and candidates; each token attends
only to its ancestors, at its depth in the path, so a candidate still reads
exactly ``[state][question][candidate]``. A state whose trie exceeds the
dense-mask budget falls back to ``question`` routing: a private KV cache per
candidate branch when one question has many candidates and a long prefix,
independent paths otherwise. Sibling candidates never attend one another.
The trained candidate-set head runs after all branch vectors exist.
"""
import copy

MODES=('auto','question','shared','path')
# Per forward pass: packed trie tokens, and the float32 ancestor mask they need.
TREE_MAX_TOKENS=8192
TREE_MAX_MASK_BYTES=64*1024*1024


def common_prefix(paths):
    # Always leave one token per candidate for extracting its final hidden state.
    limit=min(map(len, paths))-1
    for i in range(limit):
        if any(path[i]!=paths[0][i] for path in paths[1:]):return i
    return limit


def _encode_tree(engine, entries, question_states, store):
    """Encode each state that fits the budget as one trie; return the rest."""
    from agentjev.data import batch_to_device, build_tries, pack_dense_tries
    from agentjev.encoders import encode_dense
    max_tokens=getattr(engine,'tree_max_tokens',TREE_MAX_TOKENS)
    max_mask=getattr(engine,'tree_max_mask_bytes',TREE_MAX_MASK_BYTES)
    by_state={}
    for seq,loc in entries:by_state.setdefault(question_states[loc[0]],[]).append((seq,loc))
    states=list(by_state.values())
    nodes,depths,children,roots,ends=build_tries(
        [seq for group in states for seq,_ in group],[g for g,group in enumerate(states) for _ in group],len(states))
    first=[0]
    for group in states:first.append(first[-1]+len(group))
    def fits(count,width):return count*width<=max_tokens and count*width*width*4<=max_mask
    leftover=[];chunks=[]
    # Widest first, so every row of a chunk pads to the chunk's first trie.
    for g in sorted(range(len(states)),key=lambda g:-len(nodes[g])):
        if not fits(1,len(nodes[g])):leftover.extend(states[g]);continue
        if chunks and fits(len(chunks[-1])+1,len(nodes[chunks[-1][0]])):chunks[-1].append(g)
        else:chunks.append([g])
    computed=0;questions=set()
    for chunk in chunks:
        rows=[];columns=[];locations=[]
        for row,g in enumerate(chunk):
            for i in range(first[g],first[g+1]):
                rows.append(row);columns.append(ends[i]);locations.append(states[g][i-first[g]][1])
        meta=batch_to_device(pack_dense_tries([nodes[g] for g in chunk],[depths[g] for g in chunk],
            [children[g] for g in chunk],[roots[g] for g in chunk],columns,rows),engine.device)
        hidden=encode_dense(engine.model.path_encoder.backbone,meta)
        store(hidden[meta['endpoint_rows'],meta['endpoints']],locations)
        computed+=meta['input_tokens'];questions.update(qi for qi,_ in locations);del hidden
    return leftover,computed,sum(map(len,chunks)),len(questions)


def encode(engine, paths, locations, rows, mode='auto', question_states=None):
    torch=engine.torch;device=engine.device;backbone=engine.model.path_encoder.backbone
    vectors=None;mask=torch.zeros(len(rows),max(len(q['candidates']) for q in rows),dtype=torch.bool,device=device)
    def store(found,where):
        nonlocal vectors
        if vectors is None:vectors=found.new_zeros(mask.shape[0],mask.shape[1],found.shape[-1])
        qi,ci=torch.tensor(where,device=device).unbind(1)
        vectors[qi,ci]=found;mask[qi,ci]=True
    def save(hidden,chunk,offset=0):
        ends=torch.tensor([len(seq)-offset-1 for seq,_ in chunk],device=device)
        store(hidden[torch.arange(len(chunk),device=device),ends],[loc for _,loc in chunk])
    entries=list(zip(paths,locations));computed_tokens=0;shared_questions=0;shared_states=0
    if mode=='auto':
        if question_states is not None:
            entries,computed_tokens,shared_states,shared_questions=_encode_tree(engine,entries,question_states,store)
        mode='question'
    groups={}
    for seq,loc in entries:groups.setdefault(loc[0],[]).append((seq,loc))
    fallback=[]
    for entries in groups.values():
        prefix_length=common_prefix([seq for seq,_ in entries])
        use_shared=mode=='shared' or (mode=='question' and len(entries)>=8 and prefix_length>=128)
        if mode=='path' or not use_shared or prefix_length<1:
            fallback.extend(entries);continue
        prefix=torch.tensor([entries[0][0][:prefix_length]],device=device)
        encoded=backbone(input_ids=prefix,attention_mask=torch.ones_like(prefix),use_cache=True)
        root_cache=encoded.past_key_values;del encoded
        computed_tokens+=prefix_length;shared_questions+=1
        for start in range(0,len(entries),engine.path_batch):
            chunk=entries[start:start+engine.path_batch]
            suffixes=[seq[prefix_length:] for seq,_ in chunk];length=max(map(len,suffixes))
            ids=torch.full((len(chunk),length),engine.tokenizer.pad_token_id,dtype=torch.long,device=device)
            attention=torch.ones(len(chunk),prefix_length+length,dtype=torch.long,device=device)
            for i,suffix in enumerate(suffixes):
                ids[i,:len(suffix)]=torch.tensor(suffix,device=device)
                attention[i,prefix_length+len(suffix):]=0
            cache=copy.deepcopy(root_cache)
            cache.batch_repeat_interleave(len(chunk))
            hidden=backbone(input_ids=ids,attention_mask=attention,past_key_values=cache,use_cache=True).last_hidden_state
            save(hidden,chunk,prefix_length);computed_tokens+=sum(map(len,suffixes))
            del hidden,cache
        del root_cache
    for start in range(0,len(fallback),engine.path_batch):
        chunk=fallback[start:start+engine.path_batch];length=max(len(seq) for seq,_ in chunk)
        ids=torch.full((len(chunk),length),engine.tokenizer.pad_token_id,dtype=torch.long,device=device)
        attention=torch.zeros_like(ids)
        for i,(seq,_) in enumerate(chunk):
            ids[i,:len(seq)]=torch.tensor(seq,device=device);attention[i,:len(seq)]=1
        hidden=backbone(input_ids=ids,attention_mask=attention,use_cache=False).last_hidden_state
        save(hidden,chunk);computed_tokens+=sum(len(seq) for seq,_ in chunk)
        del hidden
    return vectors,mask,{'backbone_input_tokens':computed_tokens,'shared_prefix_questions':shared_questions,
                         'shared_prefix_states':shared_states}
