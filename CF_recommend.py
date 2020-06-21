import numpy as np
from word_vector import word_to_vector_sgng

def dict_slice(ori_dict, start, end):
    slice_dict = {k: ori_dict[k] for k in list(ori_dict.keys())[start:end]}
    return slice_dict

data_vector = word_to_vector_sgng()

user_id = input()
user_graph_file = open('user_graph.txt','r',encoding='utf8')
user_graph = {}
recommend = {}
for line in user_graph_file.readlines():
    user = line.split(' ')[:-1]
    user_graph[user[0]] = user[1:]

for user,words in user_graph.items():
    if user!=user_id:
        union = user_graph[user_id][:30]
        words = words[:30]
        weight = words[30:]
        intersect = []
        for i in range(len(words)):
            index = 0
            for j in range(len(union)):
                if words[i] in data_vector[1] and union[j] in data_vector[1]:
                    n1 = np.squeeze(np.asarray(data_vector[1][words[i]]))
                    n2 = np.squeeze(np.asarray(data_vector[1][union[j]]))
                    cos_distence = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
                    if cos_distence>0.6:
                        intersect.append(words[i])
                        break
                index += 1
            if index==len(union):
                union.append(words[i])
        recommend[user] = len(intersect)*1.0/len(union)
recommend = sorted(recommend.items(), key = lambda kv:(kv[1], kv[0]))[-5:]
recommend_user_id = []
for r in recommend:
    recommend_user_id.append(r[0])
print(recommend_user_id)
f = open('all_questions.txt','r',encoding='utf8')
r_articles = []
for line in f.readlines():
    line = line.split(' ')
    if line[0] in recommend_user_id:
        r_articles += line[1:]
f.close()
f=open('all_questions_info.txt','r',encoding='utf8')
questions_info = eval(f.read())
f.close()
initial_user_graph = user_graph[user_id][:30]
result = {}
for i in range(len(r_articles)):
    user_article_intersect = 0
    question_id = r_articles[i].split('/')[-1]
    if question_id in questions_info:
        topics = questions_info[question_id]['topics']
    else:
        topics = []
    for topic in topics:
        for i in range(len(initial_user_graph)):
            if topic in data_vector[1] and initial_user_graph[i] in data_vector[1]:
                v1 = np.squeeze(np.asarray(data_vector[1][topic]))
                v2 = np.squeeze(np.asarray(data_vector[1][initial_user_graph[i]]))
                cos_distence = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                if cos_distence > 0.6:
                    user_article_intersect += 1
    result[question_id] = user_article_intersect
article_result = sorted(result.items(), key = lambda kv:(kv[1], kv[0]))[-20:]
article_result.reverse()
recommend_result = []
for i in range(len(article_result)):
    if article_result[i][0] in questions_info:
        info = questions_info[article_result[i][0]]
        print("The recommend articles are:")
        print('*'*10+'recommend article {}'.format(i+1)+'*'*10)
        print("Topic:{}   Title:{}   Content:{}".format(info['topics'],info['title'],info['content']))
        recommend_result.append((article_result[i][0],info['topics'],info['title'],info['content']))