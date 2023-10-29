import spacy

nlp = spacy.load("en_core_web_trf")

sent1 = nlp("Teacher A and students need to be done.")
sent2 = nlp(
    "The teacher was taught that the graduate students need to take medicine."
)

s1 = sent1[3:4]

s2 = sent2[5:8]

print(s1)
print(s2)
print(s1.similarity(s2))

sent3 = nlp("As for Anna, things could change. She did not like it.")
s1 = sent3[2:3]
s2 = sent3[8:9]
print(s1)
print(s2)
print(s1.similarity(s2))
