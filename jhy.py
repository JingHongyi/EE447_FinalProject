from PIL import Image
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
image = np.array(Image.open("person.png"))
f1 = open('user_graph.txt','r',encoding='utf8')
user_id = input()
cloud_words = {}
for line in f1.readlines():
    content = line.split(' ')
    if content[0]==user_id:
        for i in range(30):
            cloud_words[content[1+i]] = float(content[31+i])
wc=WordCloud(scale=10,font_path='simfang.ttf',background_color='white',mask=image)
wc.generate_from_frequencies(cloud_words)
wc.to_file("./picture/picture_{}_wordcloud.png".format(user_id))
plt.figure(figsize=(10,10),dpi=100)
plt.imshow(wc)