from nltk.corpus import inaugural

import matplotlib.pyplot as plt
x=inaugural.words('2009-Obama.txt')
l={}
new=[]
k={}
z=set(x)

for word in z:
 
    l[word]=x.count(word)
#print(l)
from nltk.stem import PorterStemmer
ps=PorterStemmer()
for words in x:
    new.append(ps.stem(words))
p=set(new)
for w in p:
    k[w]=new.count(w)
plt.plot(k.values())
#plt.xlabel(k.keys())
k_sorted = sorted(k.items(), key=operator.itemgetter(1),reverse=True)
for word,count in k.items():
    if(count==max(k.values())):
        print(word)

print(k_sorted[0])
