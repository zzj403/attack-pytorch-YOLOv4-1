import json
'/disk2/mycode/0511models/pytorch-YOLOv4-master-unofficial/long_text_2020-07-09-20-46-42.json'
connected_domin_json_name = 'connected_domin_score_dict.json'
with open(connected_domin_json_name) as f_obj:
    score0 = json.load(f_obj)
with open('long_c_2020-07-09-20-46-33.json') as f_obj:
    score1 = json.load(f_obj)
num = 0
for (k,v) in score0.items():
    if score0[k] == score1[k]:
        pass
    else:
        num +=1
        print(k, num)
