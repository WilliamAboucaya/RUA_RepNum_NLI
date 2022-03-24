def remove_past_sentences(proposal_content, sentences_tokenizer, nlp_token_classifier):
    past_tense_tags = ["VER:impf", "VER:simp", "VER:subi"]

    sentences = sentences_tokenizer.tokenize(proposal_content)
    not_past_sentences = []

    for sentence in sentences:
        sentence_pos = nlp_token_classifier(sentence)
        if not any(token["entity"] in past_tense_tags for token in sentence_pos):
            not_past_sentences.append(sentence)

    return " ".join(not_past_sentences)


def predict_nli(premise, hypothesis, nli_tokenizer, nli_model) -> int:
    x = nli_tokenizer.encode(premise, hypothesis, return_tensors='pt', max_length=512, truncation=True)
    logits = nli_model(x)[0]
    probs = logits[:, ::].softmax(dim=1)
    return int(float(probs[:, 1]) > float(probs[:, 0]))
